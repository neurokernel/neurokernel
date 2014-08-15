#!/usr/bin/env python

"""
Core Neurokernel classes.
"""

import atexit
import collections
import numpy as np
import time

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import twiggy
import bidict

from base import BaseModule, Manager, Broker, PORT_DATA, PORT_CTRL, setup_logger

from ctx_managers import (IgnoreKeyboardInterrupt, OnKeyboardInterrupt,
                          ExceptionOnSignal, TryExceptionOnSignal)

from tools.comm import get_random_port
from tools.misc import catch_exception
from uid import uid
from pattern import Interface, Pattern
from plsel import PathLikeSelector, PortMapper
            
class Module(BaseModule):
    """
    Processing module.

    This class repeatedly executes a work method until it receives
    a quit message via its control port.

    Parameters
    ----------
    selector : str, unicode, or sequence
        Path-like selector describing the module's interface of 
        exposed ports.
    sel_gpot : str, unicode, or sequence
        Path-like selector describing the graded potential ports in the module's
        interface.
    sel_spike : str, unicode, or sequence
        Path-like selector describing the spiking ports in the module's
        interface.
    data_gpot : numpy.ndarray
        Data array to associate with graded potential ports. Array length
        must equal the number of graded potential ports in the module's interface.
    data_spike : numpy.ndarray
        Data array to associate with spiking ports. Array length
        must equal the number of spiking ports in the module's interface.
    columns : list of str
        Interface port attributes. This list must at least contain
        'interface', 'io', and 'type'.
    port_data : int
        Network port for transmitting data.
    port_ctrl : int
        Network port for controlling the module instance.
    id : str
        Module identifier. If no identifier is specified, a unique
        identifier is automatically generated.
    device : int
        GPU device to use.
    debug : bool
        Debug flag.
        
    Notes
    -----
    A module instance connected to other module instances contains a list of the
    connectivity objects that describe incoming connects and a list of
    masks that select for the neurons whose data must be transmitted to
    destination modules.
    """

    def __init__(self, selector, sel_gpot, sel_spike, data_gpot, data_spike,
                 columns=['interface', 'io', 'type'],
                 port_data=PORT_DATA, port_ctrl=PORT_CTRL,
                 id=None, device=None, debug=False):

        self.debug = debug
        self.device = device
        
        # Require several necessary attribute columns:
        assert 'interface' in columns
        assert 'io' in columns
        assert 'type' in columns

        # Generate a unique ID if none is specified:
        if id is None:
            id = uid()

        super(BaseModule, self).__init__(port_ctrl, id)

        # Logging:
        self.logger = twiggy.log.name('module %s' % self.id)

        # Data port:
        if port_data == port_ctrl:
            raise ValueError('data and control ports must differ')
        self.port_data = port_data

        # Initial network connectivity:
        self.net = 'none'

        # Create module interface given the specified ports:
        self.interface = Interface(selector, columns)

        # Set the interface ID to 0
        # we assume that a module only has one interface:
        self.interface[selector, 'interface'] = 0

        # Set port types:
        assert PathLikeSelector.is_in(sel_gpot, selector)
        assert PathLikeSelector.is_in(sel_spike, selector)
        assert PathLikeSelector.are_disjoint(sel_gpot, sel_spike)
        self.interface[sel_gpot, 'type'] = 'gpot'
        self.interface[sel_spike, 'type'] = 'spike'

        # Set up mapper between port identifiers and their associated data:
        assert len(data_gpot) == len(self.interface.gpot_ports())
        assert len(data_spike) == len(self.interface.spike_ports())
        self.data = {}
        self.data['gpot'] = data_gpot
        self.data['spike'] = data_spike
        self.pm = {}
        self.pm['gpot'] = PortMapper(self.data['gpot'], sel_gpot)
        self.pm['spike'] = PortMapper(self.data['spike'], sel_spike)

        # Patterns connecting this module instance with other modules instances.
        # Keyed on the IDs of those modules:
        self.patterns = {}

        # Each entry in pat_ints is a tuple containing the identifiers of which 
        # of a pattern's identifiers are connected to the current module (first
        # entry) and the modules to which it is connected (second entry).
        # Keyed on the IDs of those modules:
        self.pat_ints = {}

        # Dict for storing incoming data; each entry (corresponding to each
        # module that sends input to the current module) is a deque containing
        # incoming data, which in turn contains transmitted data arrays. Deques
        # are used here to accommodate situations when multiple data from a
        # single source arrive:
        self._in_data = {}

        # List for storing outgoing data; each entry is a tuple whose first
        # entry is the source or destination module ID and whose second entry is
        # the data to transmit:
        self._out_data = []

        # Dictionaries containing ports of source modules that
        # send output to this module. Must be initialized immediately before
        # an emulation begins running. Keyed on source module ID:
        self._in_port_dict = {}
        self._in_port_dict_ids = {}
        self._in_port_dict['gpot'] = {}
        self._in_port_dict['spike'] = {}

        # Dictionaries containing ports of destination modules that
        # receive input from this module. Must be initialized immediately before
        # an emulation begins running. Keyed on destination module ID:
        self._out_port_dict = {}
        self._out_port_dict_ids = {}
        self._out_port_dict['gpot'] = {}
        self._out_port_dict['spike'] = {}

        self._out_ids = []
        self._in_ids = []
        
    def _init_gpu(self):
        """
        Initialize GPU device.

        Notes
        -----
        Must be called from within the `run()` method, not from within
        `__init__()`.

        """

        if self.device == None:
            self.logger.info('no GPU specified - not initializing ')
        else:
            drv.init()
            try:
                self.gpu_ctx = drv.Device(self.device).make_context()
            except Exception as e:
                self.logger.info('_init_gpu exception: ' + e.message)
            else:
                atexit.register(self.gpu_ctx.pop)
                self.logger.info('GPU initialized')

    @property
    def N_gpot_ports(self):
        """
        Number of exposed graded-potential ports.
        """

        return len(self.interface.gpot_ports())

    @property
    def N_spike_ports(self):
        """
        Number of exposed spiking ports.
        """

        return len(self.interface.spike_ports())

    def _get_in_data(self):
        """
        Get input data from incoming transmission buffer.

        Populate the data arrays associated with a module's ports using input
        data received from other modules.
        """

        self.logger.info('retrieving from input buffer')
        # Since fan-in is not permitted, the data from all source modules
        # must necessarily map to different ports; we can therefore write each
        # of the received data to the array associated with the module's ports
        # here without worry of overwriting the data from each source module:
        for in_id in self._in_ids:
            # Check for exceptions so as to not fail on the first emulation
            # step when there is no input data to retrieve:
            try:
                
                # The first entry of `data` contains graded potential values,
                # while the second contains integer indices of the current
                # module's ports that should receive transmitted spikes:

                data = self._in_data[in_id].popleft()
            except:
                self.logger.info('no input data from [%s] retrieved' % in_id)
            else:
                self.logger.info('input data from [%s] retrieved' % in_id)

                # Assign transmitted graded potential values directly to port
                # data array:
                self.pm['gpot'].data[self._in_port_dict_ids['gpot'][in_id]] = data[0]
                # Clear all input spike port data..
                self.pm['spike'].data[self._in_port_dict_ids['spike'][in_id]] = 0
                    
                # ..and then set the port data using the transmitted
                # information about source module spikes:
                #self.pm['spike'][self.pm['spike'].inds_to_ports(data[1])] = 1
                self.pm['spike'][data[1]] = 1

    def _put_out_data(self):
        """
        Put specified output data in outgoing transmission buffer.

        Stage data from the data arrays associated with a module's ports for
        output to other modules.
        """

        self.logger.info('populating output buffer')

        # Clear output buffer before populating it:
        self._out_data = []

        # Select data that should be sent to each destination module and append
        # it to the outgoing queue:
        for out_id in self._out_ids:
            # Select graded potential data using list of 
            # graded potential ports that can transmit output:
            gpot_data = self.pm['gpot'].data[self._out_port_dict_ids['gpot'][out_id]]
                
            # Select spiking ports that can transmit output:
            out_spike_ports_all = self._out_port_dict['spike'][out_id]

            # Find those ports that have emitted a spike:
            out_spike_ports_with_spikes = \
                self.pm['spike'].get_ports_nonzero()

            # Compute the intersection of the two sets of spiking
            # ports obtained above to determine which ports the destination
            # module must be informed about:
            out_spike_ports = \
                list(set(out_spike_ports_all).intersection(out_spike_ports_with_spikes))

            # Find the input ports in the destination module's interface to
            # which the spikes emitted by the current module's spiking ports
            # must be sent:
            from_int, to_int = self.pat_ints[out_id]
            spike_data = \
                self.patterns[out_id].dest_idx(from_int, to_int, 'spike', 'spike',
                                               out_spike_ports)

            try:


                # Stage the emitted port data for transmission:
                self._out_data.append((out_id, (gpot_data, spike_data)))
            except:
                self.logger.info('no output data to [%s] sent' % out_id)
            else:
                self.logger.info('output data to [%s] sent' % out_id)
                
    def run_step(self):
        """
        Module work method.
    
        This method should be implemented to do something interesting with new 
        input port data in the module's `pm` attribute and update the attribute's
        output port data if necessary. It should not interact with any other 
        class attributes.
        """

        self.logger.info('running execution step')

    def _init_port_dicts(self):
        """
        Initial dictionaries of source/destination ports in current module.
        """

        # Extract identifiers of source ports in the current module's interface
        # for all modules receiving output from the current module:
        self._out_port_dict['gpot'] = {}
        self._out_port_dict['spike'] = {}
        self._out_port_dict_ids['gpot'] = {}
        self._out_port_dict_ids['spike'] = {}

        self._out_ids = self.out_ids
        for out_id in self._out_ids:
            self.logger.info('extracting output ports for %s' % out_id)

            # Get interfaces of pattern connecting the current module to
            # destination module `out_id`; `from_int` is connected to the
            # current module, `to_int` is connected to the other module:
            from_int, to_int = self.pat_ints[out_id]

            # Get ports in interface (`from_int`) connected to the current
            # module that are connected to the other module via the pattern:
            self._out_port_dict['gpot'][out_id] = \
                self.patterns[out_id].src_idx(from_int, to_int,
                                              'gpot', 'gpot')
            self._out_port_dict_ids['gpot'][out_id] = \
                self.pm['gpot'].ports_to_inds(self._out_port_dict['gpot'][out_id])
            self._out_port_dict['spike'][out_id] = \
                self.patterns[out_id].src_idx(from_int, to_int,
                                              'spike', 'spike')
            self._out_port_dict_ids['spike'][out_id] = \
                self.pm['spike'].ports_to_inds(self._out_port_dict['spike'][out_id])
                                                              
        # Extract identifiers of destination ports in the current module's
        # interface for all modules sending input to the current module:
        self._in_port_dict['gpot'] = {}
        self._in_port_dict['spike'] = {}
        self._in_port_dict_ids['gpot'] = {}
        self._in_port_dict_ids['spike'] = {}

        self._in_ids = self.in_ids
        for in_id in self._in_ids:
            self.logger.info('extracting input ports for %s' % in_id)

            # Get interfaces of pattern connecting the current module to
            # source module `out_id`; `to_int` is connected to the current
            # module, `from_int` is connected to the other module:
            to_int, from_int = self.pat_ints[in_id]

            # Get ports in interface (`to_int`) connected to the current
            # module that are connected to the other module via the pattern:
            self._in_port_dict['gpot'][in_id] = \
                self.patterns[in_id].dest_idx(from_int, to_int,
                                              'gpot', 'gpot')
            self._in_port_dict_ids['gpot'][in_id] = \
                self.pm['gpot'].ports_to_inds(self._in_port_dict['gpot'][in_id])
            self._in_port_dict['spike'][in_id] = \
                self.patterns[in_id].dest_idx(from_int, to_int,
                                              'spike', 'spike')
            self._in_port_dict_ids['spike'][in_id] = \
                self.pm['spike'].ports_to_inds(self._in_port_dict['spike'][in_id])
            
    def pre_run(self, *args, **kwargs):
        """
        Code to run before main module run loop.

        Code in this method will be executed after a module's process has been
        launched and all connectivity objects made available, but before the
        main run loop begins.
        """
        
        self._init_gpu()
        pass

    def run(self):
        """
        Body of process.
        """

        # Don't allow keyboard interruption of process:
        self.logger.info('starting')
        with IgnoreKeyboardInterrupt():
            # Initialize environment:
            self._init_net()
            
            # Initialize _out_port_dict and _in_port_dict attributes:
            self._init_port_dicts()

            
            # Initialize Buffer for incoming data.  Dict used to store the
            # incoming data keyed by the source module id.  Each value is a
            # queue buferring the received data:
            self._in_data = {k: collections.deque() for k in self._in_ids}

            # Perform any pre-emulation operations:
            self.pre_run()

            self.running = True
            curr_steps = 0
            while curr_steps < self._steps:
                self.logger.info('execution step: %s' % curr_steps)
                # If the debug flag is set, don't catch exceptions so that
                # errors will lead to visible failures:
                if self.debug:

                    # Get transmitted input data for processing:
                    self._get_in_data()

                    # Run the processing step:
                    self.run_step()

                    # Stage generated output data for transmission to other
                    # modules:
                    self._put_out_data()

                    # Synchronize:
                    self._sync()

                else:

                    # Get transmitted input data for processing:
                    catch_exception(self._get_in_data, self.logger.info)

                    # Run the processing step:
                    catch_exception(self.run_step, self.logger.info)

                    # Stage generated output data for transmission to other
                    # modules:
                    catch_exception(self._put_out_data, self.logger.info)

                    # Synchronize:
                    catch_exception(self._sync, self.logger.info)

                # Exit run loop when a quit signal has been received:
                if not self.running:
                    self.logger.info('run loop stopped')
                    break

                curr_steps += 1

            # Perform any post-emulation operations:
            self.post_run()

            # Shut down the control handler and inform the manager that the
            # module has shut down:
            self._ctrl_stream_shutdown()
            ack = 'shutdown'
            self.sock_ctrl.send(ack)
            self.logger.info('sent to manager: %s' % ack)
                
        self.logger.info('exiting')
        

if __name__ == '__main__':
    import time

    class MyModule(Module):
        """
        Example of derived module class.
        """

        def __init__(self, sel, 
                     sel_in_gpot, sel_in_spike,
                     sel_out_gpot, sel_out_spike,
                     data_gpot, data_spike,
                     columns=['interface', 'io', 'type'],
                     port_data=PORT_DATA, port_ctrl=PORT_CTRL,
                     id=None, device=None):                     
            super(MyModule, self).__init__(sel, ','.join([sel_in_gpot, 
                                                          sel_out_gpot]),
                                           ','.join([sel_in_spike,
                                                     sel_out_spike]),
                                           data_gpot, data_spike,
                                           columns, port_data, port_ctrl,
                                           id, None, True)

            
            assert PathLikeSelector.is_in(sel_in_gpot, sel)
            assert PathLikeSelector.is_in(sel_out_gpot, sel)
            assert PathLikeSelector.are_disjoint(sel_in_gpot, sel_out_gpot)
            assert PathLikeSelector.is_in(sel_in_spike, sel)
            assert PathLikeSelector.is_in(sel_out_spike, sel)
            assert PathLikeSelector.are_disjoint(sel_in_spike, sel_out_spike)

            self.interface[sel_in_gpot, 'io', 'type'] = ['in', 'gpot']
            self.interface[sel_out_gpot, 'io', 'type'] = ['out', 'gpot']
            self.interface[sel_in_spike, 'io', 'type'] = ['in', 'spike']
            self.interface[sel_out_spike, 'io', 'type'] = ['out', 'spike']

        def run_step(self):
            super(MyModule, self).run_step()

            # Do something with input graded potential data:
            in_gpot_ports = self.interface.in_ports().gpot_ports().to_tuples()
            self.logger.info('input gpot port data: '+str(self.pm['gpot'][in_gpot_ports]))

            # Do something with input spike data:
            in_spike_ports = self.interface.in_ports().spike_ports().to_tuples()
            self.logger.info('input spike port data: '+str(self.pm['spike'][in_spike_ports]))

            # Output random graded potential data:
            out_gpot_ports = self.interface.out_ports().gpot_ports().to_tuples()
            self.pm['gpot'][out_gpot_ports] = \
                    np.random.rand(len(out_gpot_ports))
            
            # Randomly select output ports to emit spikes:
            out_spike_ports = self.interface.out_ports().spike_ports().to_tuples()
            self.pm['spike'][out_spike_ports] = \
                    np.random.randint(0, 2, len(out_spike_ports))

        def run(self):

            # Make every class instance generate a different pseudorandom sequence:
            np.random.seed(id(self))
            super(MyModule, self).run()

    def emulate(n, steps):
        assert(n>1)
        n = str(n)
        # Set up emulation:
        man = Manager(get_random_port(), get_random_port())
        man.add_brok()

        m1_int_sel_in_gpot = '/a/in/gpot0,/a/in/gpot1'
        m1_int_sel_out_gpot = '/a/out/gpot0,/a/out/gpot1'
        m1_int_sel_in_spike = '/a/in/spike0,/a/in/spike1'
        m1_int_sel_out_spike = '/a/out/spike0,/a/out/spike1'
        m1_int_sel = ','.join([m1_int_sel_in_gpot, m1_int_sel_out_gpot,
                               m1_int_sel_in_spike, m1_int_sel_out_spike])
        N1_gpot = PathLikeSelector.count_ports(','.join([m1_int_sel_in_gpot,
                                                         m1_int_sel_out_gpot]))
        N1_spike = PathLikeSelector.count_ports(','.join([m1_int_sel_in_spike,
                                                          m1_int_sel_out_spike]))
        m1 = MyModule(m1_int_sel, 
                      m1_int_sel_in_gpot, m1_int_sel_in_spike,
                      m1_int_sel_out_gpot, m1_int_sel_out_spike,
                      np.zeros(N1_gpot, np.float64),
                      np.zeros(N1_spike, int), ['interface', 'io', 'type'],
                      man.port_data, man.port_ctrl, 'm1')
        man.add_mod(m1)

        m2_int_sel_in_gpot = '/b/in/gpot0,/b/in/gpot1'
        m2_int_sel_out_gpot = '/b/out/gpot0,/b/out/gpot1'
        m2_int_sel_in_spike = '/b/in/spike0,/b/in/spike1'
        m2_int_sel_out_spike = '/b/out/spike0,/b/out/spike1'
        m2_int_sel = ','.join([m2_int_sel_in_gpot, m2_int_sel_out_gpot,
                               m2_int_sel_in_spike, m2_int_sel_out_spike])
        N2_gpot = PathLikeSelector.count_ports(','.join([m2_int_sel_in_gpot,
                                                         m2_int_sel_out_gpot]))
        N2_spike = PathLikeSelector.count_ports(','.join([m2_int_sel_in_spike,
                                                          m2_int_sel_out_spike]))
        m2 = MyModule(m2_int_sel,
                      m2_int_sel_in_gpot, m2_int_sel_in_spike,
                      m2_int_sel_out_gpot, m2_int_sel_out_spike,
                      np.zeros(N2_gpot, np.float64),
                      np.zeros(N2_spike, int), ['interface', 'io', 'type'],
                      man.port_data, man.port_ctrl, 'm2')
                                             
        # Make sure that all ports in the patterns' interfaces are set so 
        # that they match those of the modules:
        pat12 = Pattern(m1_int_sel, m2_int_sel)
        pat12.interface[m1_int_sel_out_gpot] = [0, 'in', 'gpot']
        pat12.interface[m1_int_sel_in_gpot] = [0, 'out', 'gpot']
        pat12.interface[m1_int_sel_out_spike] = [0, 'in', 'spike']
        pat12.interface[m1_int_sel_in_spike] = [0, 'out', 'spike']
        pat12.interface[m2_int_sel_in_gpot] = [1, 'out', 'gpot']
        pat12.interface[m2_int_sel_out_gpot] = [1, 'in', 'gpot']
        pat12.interface[m2_int_sel_in_spike] = [1, 'out', 'spike']
        pat12.interface[m2_int_sel_out_spike] = [1, 'in', 'spike']
        pat12['/a/out/gpot0', '/b/in/gpot0'] = 1
        pat12['/a/out/gpot1', '/b/in/gpot1'] = 1
        pat12['/b/out/gpot0', '/a/in/gpot0'] = 1
        pat12['/b/out/gpot1', '/a/in/gpot1'] = 1
        pat12['/a/out/spike0', '/b/in/spike0'] = 1
        pat12['/a/out/spike1', '/b/in/spike1'] = 1
        pat12['/b/out/spike0', '/a/in/spike0'] = 1
        pat12['/b/out/spike1', '/a/in/spike1'] = 1
        man.connect(m1, m2, pat12, 0, 1)

        # To set the emulation to exit after executing a fixed number of steps,
        # start it as follows and remove the sleep statement:
        man.start(steps=steps)
        # man.start()
        # time.sleep(2)
        man.stop()
        return m1
        
    # Set up logging:
    logger = setup_logger(screen=False)
    steps = 100

    # Emulation 1
    start_time = time.time()
    size = 2
    m1 = emulate(size, steps)
    print('Simulation of size {} complete: Duration {} seconds'.format(
        size, time.time() - start_time))
    # Emulation 2
    start_time = time.time()
    size = 100
    emulate(size, steps)
    print('Simulation of size {} complete: Duration {} seconds'.format(
        size, time.time() - start_time))
    logger.info('all done')
