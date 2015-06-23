#!/usr/bin/env python

"""
Core Neurokernel classes.
"""

import atexit
import collections
import time

import bidict
import msgpack
import numpy as np
import twiggy

from mixins import LoggerMixin
from base import BaseModule, BaseManager, Broker, \
    PORT_DATA, PORT_CTRL, PORT_TIME

from ctx_managers import (IgnoreKeyboardInterrupt, OnKeyboardInterrupt,
                          ExceptionOnSignal, TryExceptionOnSignal)
from tools.logging import setup_logger
from tools.comm import get_random_port
from tools.misc import catch_exception
from uid import uid
from pattern import Interface, Pattern
from plsel import SelectorMethods, BasePortMapper, PortMapper

class Module(BaseModule):
    """
    Processing module.

    This class repeatedly executes a work method until it receives
    a quit message via its control port.

    Parameters
    ----------
    sel : str, unicode, or sequence
        Path-like selector describing the module's interface of
        exposed ports.
    sel_in : str, unicode, or sequence
        Selector describing all input ports in the module's interface.
    sel_out : str, unicode, or sequence
        Selector describing all input ports in the module's interface.
    sel_gpot : str, unicode, or sequence
        Selector describing all graded potential ports in the module's
        interface.
    sel_spike : str, unicode, or sequence
        Selector describing all spiking ports in the module's interface.
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
    time_sync : bool
        Time synchronization flag. When True, debug messages are not emitted during
        module synchronization and the time taken to receive all incoming data is 
        computed.
        
    Notes
    -----
    A module instance connected to other module instances contains a list of the
    connectivity objects that describe incoming connects and a list of
    masks that select for the neurons whose data must be transmitted to
    destination modules.
    """

    def __init__(self, sel, sel_in, sel_out, 
                 sel_gpot, sel_spike,
                 data_gpot, data_spike,
                 columns=['interface', 'io', 'type'],
                 port_data=PORT_DATA, port_ctrl=PORT_CTRL, port_time=PORT_TIME,
                 id=None, device=None, debug=False, time_sync=False):

        self.debug = debug
        self.time_sync = time_sync
        self.device = device

        # Require several necessary attribute columns:
        assert 'interface' in columns
        assert 'io' in columns
        assert 'type' in columns

        # Generate a unique ID if none is specified:
        if id is None:
            id = uid()

        # Call super for BaseModule rather than Module because most of the
        # functionality of the former's constructor must be overridden in any case:
        super(BaseModule, self).__init__(port_ctrl, id)

        # Reformat logger name:
        LoggerMixin.__init__(self, 'mod %s' % self.id)

        # Data port:
        if port_data == port_ctrl:
            raise ValueError('data and control ports must differ')
        self.port_data = port_data
        if port_time == port_ctrl or port_time == port_data:
            raise ValueError('time port must differ from data and control ports')
        self.port_time = port_time

        # Initial network connectivity:
        self.net = 'none'

        # Create module interface given the specified ports:
        self.interface = Interface(sel, columns)

        # Set the interface ID to 0
        # we assume that a module only has one interface:
        self.interface[sel, 'interface'] = 0

        # Set port types:
        assert SelectorMethods.is_in(sel_in, sel)
        assert SelectorMethods.is_in(sel_out, sel)
        assert SelectorMethods.are_disjoint(sel_in, sel_out)
        self.interface[sel_in, 'io'] = 'in'
        self.interface[sel_out, 'io'] = 'out'
        assert SelectorMethods.is_in(sel_gpot, sel)
        assert SelectorMethods.is_in(sel_spike, sel)
        assert SelectorMethods.are_disjoint(sel_gpot, sel_spike)
        self.interface[sel_gpot, 'type'] = 'gpot'
        self.interface[sel_spike, 'type'] = 'spike'

        # Set up mapper between port identifiers and their associated data:
        assert len(data_gpot) == len(self.interface.gpot_ports())
        assert len(data_spike) == len(self.interface.spike_ports())
        self.data = {}
        self.data['gpot'] = data_gpot
        self.data['spike'] = data_spike
        self.pm = {}
        self.pm['gpot'] = PortMapper(sel_gpot, self.data['gpot'])
        self.pm['spike'] = PortMapper(sel_spike, self.data['spike'])

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
            self.log_info('no GPU specified - not initializing ')
        else:

            # Import pycuda.driver here so as to facilitate the
            # subclassing of Module to create pure Python LPUs that don't use GPUs:
            import pycuda.driver as drv
            drv.init()
            try:
                self.gpu_ctx = drv.Device(self.device).make_context()
            except Exception as e:
                self.log_info('_init_gpu exception: ' + e.message)
            else:
                atexit.register(self.gpu_ctx.pop)
                self.log_info('GPU initialized')

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

        if self.net in ['none', 'ctrl']:
            self.log_info('not retrieving from input buffer')
        else:
            self.log_info('retrieving from input buffer')

            # Since fan-in is not permitted, the data from all source modules
            # must necessarily map to different ports; we can therefore write each
            # of the received data to the array associated with the module's ports
            # here without worry of overwriting the data from each source module:
            for in_id in self._in_ids:
                # Check for exceptions so as to not fail on the first emulation
                # step when there is no input data to retrieve:
                try:

                    # The first entry of `data` contains graded potential values,
                    # while the second contains spiking port values (i.e., 0 or
                    # 1):
                    data = self._in_data[in_id].popleft()
                except:
                    self.log_info('no input data from [%s] retrieved' % in_id)
                else:
                    self.log_info('input data from [%s] retrieved' % in_id)

                    # Assign transmitted values directly to port data array:
                    if len(self._in_port_dict_ids['gpot'][in_id]) and data[0] is not None:
                        self.pm['gpot'].set_by_inds(self._in_port_dict_ids['gpot'][in_id], data[0])
                    if len(self._in_port_dict_ids['spike'][in_id]) and data[1] is not None:
                        self.pm['spike'].set_by_inds(self._in_port_dict_ids['spike'][in_id], data[1])

    def _put_out_data(self):
        """
        Put specified output data in outgoing transmission buffer.

        Stage data from the data arrays associated with a module's ports for
        output to other modules.

        Notes
        -----
        The output spike port selection algorithm could probably be made faster.
        """

        if self.net in ['none', 'ctrl']:
            self.log_info('not populating output buffer')
        else:
            self.log_info('populating output buffer')

            # Clear output buffer before populating it:
            self._out_data = []

            # Select data that should be sent to each destination module and append
            # it to the outgoing queue:
            for out_id in self._out_ids:
                # Select port data using list of graded potential ports that can
                # transmit output:
                if len(self._out_port_dict_ids['gpot'][out_id]):
                    gpot_data = \
                        self.pm['gpot'].get_by_inds(self._out_port_dict_ids['gpot'][out_id])
                else:
                    gpot_data = None
                if len(self._out_port_dict_ids['spike'][out_id]):
                    spike_data = \
                        self.pm['spike'].get_by_inds(self._out_port_dict_ids['spike'][out_id])
                else:
                    spike_data = None

                # Attempt to stage the emitted port data for transmission:            
                try:
                    self._out_data.append((out_id, (gpot_data, spike_data)))
                except:
                    self.log_info('no output data to [%s] sent' % out_id)
                else:
                    self.log_info('output data to [%s] sent' % out_id)
                
    def run_step(self):
        """
        Module work method.
    
        This method should be implemented to do something interesting with new 
        input port data in the module's `pm` attribute and update the attribute's
        output port data if necessary. It should not interact with any other 
        class attributes.
        """

        self.log_info('running execution step')

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
            self.log_info('extracting output ports for %s' % out_id)

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
            self.log_info('extracting input ports for %s' % in_id)

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
        main run loop begins. Initialization routines (such as GPU
        initialization) should be performed in this method.
        """

        pass

    def run(self):
        """
        Body of process.
        """

        # Don't allow keyboard interruption of process:
        self.log_info('starting')
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
            self.steps = 0
            if self.time_sync:
                self.sock_time.send(msgpack.packb((self.id, self.steps, 'start',
                                                   time.time())))
                self.log_info('sent start time to master')

            # Counter for number of steps between synchronizations:
            steps_since_sync = 0
            while self.steps < self.max_steps:
                self.log_info('execution step: %s/%s' % (self.steps, self.max_steps))

                # If the debug flag is set, don't catch exceptions so that
                # errors will lead to visible failures:
                if self.debug:

                    # Run the processing step:
                    self.run_step()

                    # Do post-processing:
                    self.post_run_step()

                    # Synchronize:
                    if steps_since_sync == self.sync_period:
                        self._sync()
                        steps_since_sync = 0
                    else:
                        self.log_info('skipping sync (%s/%s)' % \
                                      (steps_since_sync, self.sync_period))
                        steps_since_sync += 1

                else:

                    # Run the processing step:
                    catch_exception(self.run_step, self.log_info)

                    # Do post processing:
                    catch_exception(self.post_run_step, self.log_info)

                    # Synchronize:
                    if steps_since_sync == self.sync_period:
                        catch_exception(self._sync, self.log_info)
                        steps_since_sync = 0
                    else:
                        self.log_info('skipping sync (%s/%s)' % \
                                      (steps_since_sync, self.sync_period))
                        steps_since_sync += 1

                # Exit run loop when a quit signal has been received:
                if not self.running:
                    self.log_info('run loop stopped')
                    break

                self.steps += 1
            if self.time_sync:
                self.sock_time.send(msgpack.packb((self.id, self.steps, 'stop',
                                                   time.time())))
                self.log_info('sent stop time to master')
            self.log_info('maximum number of steps reached')

            # Perform any post-emulation operations:
            self.post_run()

            # Shut down the control handler and inform the manager that the
            # module has shut down:
            self._ctrl_stream_shutdown()
            ack = 'shutdown'
            self.sock_ctrl.send(ack)
            self.log_info('sent to manager: %s' % ack)
                
        self.log_info('exiting')

class Manager(BaseManager):
    """
    Module manager.

    Instantiates, connects, starts, and stops modules comprised by an 
    emulation.

    Parameters
    ----------
    port_data : int
        Port to use for communication with modules.
    port_ctrl : int
        Port used to control modules.

    Attributes
    ----------
    brokers : dict
        Communication brokers. Keyed by broker object ID.
    modules : dict
        Module instances. Keyed by module object ID.
    routing_table : routing_table.RoutingTable
        Table of data transmission connections between modules.
    """ 

    def connect(self, m_0, m_1, pat, int_0=0, int_1=1, compat_check=True):
        """
        Connect two module instances with a Pattern instance.

        Parameters
        ----------
        m_0, m_1 : BaseModule
            Module instances to connect.
        pat : Pattern
            Pattern instance.
        int_0, int_1 : int
            Which of the pattern's interfaces to connect to `m_0` and `m_1`,
            respectively.
        compat_check : bool
            Check whether the interfaces of the specified modules
            are compatible with the specified pattern. This option is provided
            because compatibility checking can be expensive.        
        """

        assert isinstance(m_0, BaseModule) and isinstance(m_1, BaseModule)
        assert isinstance(pat, Pattern)
        assert int_0 in pat.interface_ids and int_1 in pat.interface_ids

        self.log_info('connecting modules {0} and {1}'
                         .format(m_0.id, m_1.id))

        # Check whether the interfaces exposed by the modules and the
        # pattern share compatible subsets of ports:
        if compat_check:
            self.log_info('checking compatibility of modules {0} and {1} and'
                             ' assigned pattern'.format(m_0.id, m_1.id))
            assert m_0.interface.is_compatible(0, pat.interface, int_0, True)
            assert m_1.interface.is_compatible(0, pat.interface, int_1, True)

        # Find the ports common to each of the pattern's interfaces and the
        # respective interfaces of the modules connected to them; these two sets
        # of ports should be disjoint:
        common_spike_ports_0 = \
            m_0.interface.get_common_ports(0, pat.interface, int_0, 'spike')
        common_spike_ports_1 = \
            m_1.interface.get_common_ports(0, pat.interface, int_1, 'spike')
        assert set(common_spike_ports_0).isdisjoint(common_spike_ports_1)

        # Set the mappings between port identifiers and integer indices in the
        # pattern's interfaces to conform to those of the connected modules'
        # respective interfaces. Note that only one port mapper instance is
        # needed to store the mappings for both of the pattern's interfaces
        # because the respective sets of ports in the interfaces are disjoint:
        pat.interface.pm['spike'] = \
            BasePortMapper(common_spike_ports_0+common_spike_ports_1,
            np.concatenate((m_0.pm['spike'].get_map(common_spike_ports_0),
                            m_1.pm['spike'].get_map(common_spike_ports_1))))

        # Add the module and pattern instances to the internal dictionaries of
        # the manager instance if they are not already there:
        if m_0.id not in self.modules:
            self.add_mod(m_0)
        if m_1.id not in self.modules:
            self.add_mod(m_1)

        # Make the timing listener aware of the module IDs:
        self.time_listener.add(m_0.id)
        self.time_listener.add(m_1.id)

        # Pass the pattern to the modules being connected:
        self.log_info('passing connection pattern to modules {0} and {1}'.format(m_0.id, m_1.id))
        m_0.connect(m_1, pat, int_0, int_1, compat_check)
        m_1.connect(m_0, pat, int_1, int_0, compat_check)

        # Update the routing table:
        self.log_info('updating routing table')
        if pat.is_connected(0, 1):
            self.routing_table[m_0.id, m_1.id] = 1
        if pat.is_connected(1, 0):
            self.routing_table[m_1.id, m_0.id] = 1

        self.log_info('connected modules {0} and {1}'.format(m_0.id, m_1.id))

if __name__ == '__main__':
    import time

    class MyModule(Module):
        """
        Example of derived module class.
        """

        def run_step(self):
            super(MyModule, self).run_step()

            # Do something with input graded potential data:
            in_gpot_ports = self.interface.in_ports().gpot_ports().to_tuples()
            self.log_info('input gpot port data: '+str(self.pm['gpot'][in_gpot_ports]))

            # Do something with input spike data:
            in_spike_ports = self.interface.in_ports().spike_ports().to_tuples()
            self.log_info('input spike port data: '+str(self.pm['spike'][in_spike_ports]))

            # Output random graded potential data:
            out_gpot_ports = self.interface.out_ports().gpot_ports().to_tuples()
            self.pm['gpot'][out_gpot_ports] = \
                    np.random.rand(len(out_gpot_ports))

            # Randomly select output ports to emit spikes:
            out_spike_ports = self.interface.out_ports().spike_ports().to_tuples()
            self.pm['spike'][out_spike_ports] = \
                    np.random.randint(0, 2, len(out_spike_ports))

    def emulate(n, steps):
        assert(n>1)
        n = str(n)

        # Set up emulation:
        man = Manager(get_random_port(), get_random_port(), get_random_port())
        man.add_brok()

        m1_int_sel_in_gpot = '/a/in/gpot0,/a/in/gpot1'
        m1_int_sel_out_gpot = '/a/out/gpot0,/a/out/gpot1'
        m1_int_sel_in_spike = '/a/in/spike0,/a/in/spike1'
        m1_int_sel_out_spike = '/a/out/spike0,/a/out/spike1'
        m1_int_sel = ','.join([m1_int_sel_in_gpot, m1_int_sel_out_gpot,
                               m1_int_sel_in_spike, m1_int_sel_out_spike])
        m1_int_sel_in = ','.join([m1_int_sel_in_gpot, m1_int_sel_in_spike])
        m1_int_sel_out = ','.join([m1_int_sel_out_gpot, m1_int_sel_out_spike])
        m1_int_sel_gpot = ','.join([m1_int_sel_in_gpot, m1_int_sel_out_gpot])
        m1_int_sel_spike = ','.join([m1_int_sel_in_spike, m1_int_sel_out_spike])
        N1_gpot = SelectorMethods.count_ports(m1_int_sel_gpot)
        N1_spike = SelectorMethods.count_ports(m1_int_sel_spike)
        m1 = MyModule(m1_int_sel,
                      m1_int_sel_in, m1_int_sel_out,
                      m1_int_sel_gpot, m1_int_sel_spike,
                      np.zeros(N1_gpot, np.float64),
                      np.zeros(N1_spike, int), ['interface', 'io', 'type'],
                      man.port_data, man.port_ctrl, man.port_time, 'm1', None,
                      False, True)
        man.add_mod(m1)

        m2_int_sel_in_gpot = '/b/in/gpot0,/b/in/gpot1'
        m2_int_sel_out_gpot = '/b/out/gpot0,/b/out/gpot1'
        m2_int_sel_in_spike = '/b/in/spike0,/b/in/spike1'
        m2_int_sel_out_spike = '/b/out/spike0,/b/out/spike1'
        m2_int_sel = ','.join([m2_int_sel_in_gpot, m2_int_sel_out_gpot,
                               m2_int_sel_in_spike, m2_int_sel_out_spike])
        m2_int_sel_in = ','.join([m2_int_sel_in_gpot, m2_int_sel_in_spike])
        m2_int_sel_out = ','.join([m2_int_sel_out_gpot, m2_int_sel_out_spike])
        m2_int_sel_gpot = ','.join([m2_int_sel_in_gpot, m2_int_sel_out_gpot])
        m2_int_sel_spike = ','.join([m2_int_sel_in_spike, m2_int_sel_out_spike])
        N2_gpot = SelectorMethods.count_ports(m2_int_sel_gpot)
        N2_spike = SelectorMethods.count_ports(m2_int_sel_spike),
        m2 = MyModule(m2_int_sel,
                      m2_int_sel_in, m2_int_sel_out,
                      m2_int_sel_gpot, m2_int_sel_spike,
                      np.zeros(N2_gpot, np.float64),
                      np.zeros(N2_spike, int), ['interface', 'io', 'type'],
                      man.port_data, man.port_ctrl, man.port_time, 'm2', None,
                      False, True)
        man.add_mod(m2)

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
    logger = setup_logger(screen=True, multiline=True)
    steps = 100

    # Emulation 1
    start_time = time.time()
    size = 2
    m1 = emulate(size, steps)
    print('Simulation of size {} complete: Duration {} seconds'.format(
        size, time.time() - start_time))
    # Emulation 2
    # start_time = time.time()
    # size = 100
    # emulate(size, steps)
    # print('Simulation of size {} complete: Duration {} seconds'.format(
    #     size, time.time() - start_time))
    # logger.info('all done')
