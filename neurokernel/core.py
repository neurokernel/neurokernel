#!/usr/bin/env python

"""
Core Neurokernel classes.
"""

import atexit
import time

from future.utils import iteritems
import bidict
from mpi4py import MPI
import numpy as np
import twiggy

from .ctx_managers import IgnoreKeyboardInterrupt, OnKeyboardInterrupt, \
     ExceptionOnSignal, TryExceptionOnSignal
from .mixins import LoggerMixin
from . import mpi
from .tools.gpu import bufint
from .tools.logging import setup_logger
from .tools.misc import catch_exception, dtype_to_mpi, renumber_in_order
from .tools.mpi import MPIOutput
from .pattern import Interface, Pattern
from .plsel import Selector, SelectorMethods
from .pm import BasePortMapper, PortMapper
from .routing_table import RoutingTable
from .uid import uid

CTRL_TAG = 1

# MPI tags for distinguishing messages associated with different port types:
GPOT_TAG = CTRL_TAG+1
SPIKE_TAG = CTRL_TAG+2

class Module(mpi.Worker):
    """
    Processing module.

    This class repeatedly executes a work method until it receives a
    quit message via its control network port.

    Parameters
    ----------
    sel : str, unicode, or sequence
        Path-like selector describing the module's interface of
        exposed ports.
    sel_in, sel_out, sel_gpot, sel_spike : str, unicode, or sequence
        Selectors respectively describing all input, output, graded potential,
        and spiking ports in the module's interface.
    data_gpot, data_spike : numpy.ndarray
        Data arrays associated with the graded potential and spiking ports in
        the . Array length must equal the number
        of ports in a module's interface.
    columns : list of str
        Interface port attributes.
        Network port for controlling the module instance.
    ctrl_tag, gpot_tag, spike_tag : int
        MPI tags that respectively identify messages containing control data,
        graded potential port values, and spiking port values transmitted to
        worker nodes.
    id : str
        Module identifier. If no identifier is specified, a unique
        identifier is automatically generated.
    device : int
        GPU device to use. May be set to None if the module does not perform
        GPU processing.
    routing_table : neurokernel.routing_table.RoutingTable
        Routing table describing data connections between modules. If no routing
        table is specified, the module will be executed in isolation.
    rank_to_id : bidict.bidict
        Mapping between MPI ranks and module object IDs.
    debug : bool
        Debug flag. When True, exceptions raised during the work method
        are not be suppressed.
    time_sync : bool
        Time synchronization flag. When True, debug messages are not emitted
        during module synchronization and the time taken to receive all incoming
        data is computed.

    Attributes
    ----------
    interface : Interface
        Object containing information about a module's ports.
    pm : dict
        `pm['gpot']` and `pm['spike']` are instances of neurokernel.pm.PortMapper that
        map a module's ports to the contents of the values in `data`.
    data : dict
        `data['gpot']` and `data['spike']` are arrays of data associated with
        a module's graded potential and spiking ports.
    """

    def __init__(self, sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns=['interface', 'io', 'type'],
                 ctrl_tag=CTRL_TAG, gpot_tag=GPOT_TAG, spike_tag=SPIKE_TAG,
                 id=None, device=None,
                 routing_table=None, rank_to_id=None,
                 debug=False, time_sync=False):

        super(Module, self).__init__(ctrl_tag)
        self.debug = debug
        self.time_sync = time_sync
        self.device = device

        self._gpot_tag = gpot_tag
        self._spike_tag = spike_tag

        # Require several necessary attribute columns:
        if 'interface' not in columns:
            raise ValueError('interface column required')
        if 'io' not in columns:
            raise ValueError('io column required')
        if 'type' not in columns:
            raise ValueError('type column required')

        # Manually register the file close method associated with MPIOutput
        # so that it is called by atexit before MPI.Finalize() (if the file is
        # closed after MPI.Finalize() is called, an error will occur):
        for k, v in iteritems(twiggy.emitters):
             if isinstance(v._output, MPIOutput):
                 atexit.register(v._output.close)

        # Ensure that the input and output port selectors respectively
        # select mutually exclusive subsets of the set of all ports exposed by
        # the module:
        if not SelectorMethods.is_in(sel_in, sel):
            raise ValueError('input port selector not in selector of all ports')
        if not SelectorMethods.is_in(sel_out, sel):
            raise ValueError('output port selector not in selector of all ports')
        if not SelectorMethods.are_disjoint(sel_in, sel_out):
            raise ValueError('input and output port selectors not disjoint')

        # Ensure that the graded potential and spiking port selectors
        # respectively select mutually exclusive subsets of the set of all ports
        # exposed by the module:
        if not SelectorMethods.is_in(sel_gpot, sel):
            raise ValueError('gpot port selector not in selector of all ports')
        if not SelectorMethods.is_in(sel_spike, sel):
            raise ValueError('spike port selector not in selector of all ports')
        if not SelectorMethods.are_disjoint(sel_gpot, sel_spike):
            raise ValueError('gpot and spike port selectors not disjoint')

        # Save routing table and mapping between MPI ranks and module IDs:
        self.routing_table = routing_table
        self.rank_to_id = rank_to_id

        # Generate a unique ID if none is specified:
        if id is None:
            self.id = uid()
        else:

            # If a unique ID was specified and the routing table is not empty
            # (i.e., there are connections between multiple modules), the id
            # must be a node in the routing table:
            if routing_table is not None and len(routing_table.ids) and \
                    not routing_table.has_node(id):
                raise ValueError('routing table must contain specified '
                                 'module ID: {}'.format(id))
            self.id = id

        # Reformat logger name:
        LoggerMixin.__init__(self, 'mod %s' % self.id)

        # Create module interface given the specified ports:
        self.interface = Interface(sel, columns)

        # Set the interface ID to 0; we assume that a module only has one interface:
        self.interface[sel, 'interface'] = 0

        # Set the port attributes:
        self.interface[sel_in, 'io'] = 'in'
        self.interface[sel_out, 'io'] = 'out'
        self.interface[sel_gpot, 'type'] = 'gpot'
        self.interface[sel_spike, 'type'] = 'spike'

        # Find the input and output ports:
        self.in_ports = self.interface.in_ports().to_tuples()
        self.out_ports = self.interface.out_ports().to_tuples()

        # Find the graded potential and spiking ports:
        self.gpot_ports = self.interface.gpot_ports().to_tuples()
        self.spike_ports = self.interface.spike_ports().to_tuples()

        self.in_gpot_ports = self.interface.in_ports().gpot_ports().to_tuples()
        self.in_spike_ports = self.interface.in_ports().spike_ports().to_tuples()
        self.out_gpot_ports = self.interface.out_ports().gpot_ports().to_tuples()
        self.out_spike_ports = self.interface.out_ports().spike_ports().to_tuples()

        # Set up mapper between port identifiers and their associated data:
        if len(data_gpot) != len(self.gpot_ports):
            raise ValueError('incompatible gpot port data array length')
        if len(data_spike) != len(self.spike_ports):
            raise ValueError('incompatible spike port data array length')
        self.data = {}
        self.data['gpot'] = data_gpot
        self.data['spike'] = data_spike
        self.pm = {}
        self.pm['gpot'] = PortMapper(sel_gpot, self.data['gpot'], make_copy=False)
        self.pm['spike'] = PortMapper(sel_spike, self.data['spike'], make_copy=False)

        # MPI Request object for resolving asynchronous transfers:
        self.req = MPI.Request()

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
                self.log_info('GPU %s initialized' % self.device)

    def _init_port_dicts(self):
        """
        Initial dictionaries of source/destination ports in current module.
        """

        # Extract identifiers of source ports in the current module's interface
        # for all modules receiving output from the current module:
        self._out_port_dict_ids = {}
        self._out_port_dict_ids['gpot'] = {}
        self._out_port_dict_ids['spike'] = {}

        self._out_ids = self.routing_table.dest_ids(self.id)
        self._out_ranks = [self.rank_to_id.inv[i] for i in self._out_ids]
        for out_id in self._out_ids:
            self.log_info('extracting output ports for %s' % out_id)

            # Get interfaces of pattern connecting the current module to
            # destination module `out_id`; `int_0` is connected to the
            # current module, `int_1` is connected to the other module:
            pat = self.routing_table[self.id, out_id]['pattern']
            int_0 = self.routing_table[self.id, out_id]['int_0']
            int_1 = self.routing_table[self.id, out_id]['int_1']

            # Get ports in interface (`int_0`) connected to the current
            # module that are connected to the other module via the pattern:
            self._out_port_dict_ids['gpot'][out_id] = \
                self.pm['gpot'].ports_to_inds(pat.src_idx(int_0, int_1, 'gpot', 'gpot'))
            self._out_port_dict_ids['spike'][out_id] = \
                self.pm['spike'].ports_to_inds(pat.src_idx(int_0, int_1, 'spike', 'spike'))

        # Extract identifiers of destination ports in the current module's
        # interface for all modules sending input to the current module:
        self._in_port_dict_ids = {}
        self._in_port_dict_ids['gpot'] = {}
        self._in_port_dict_ids['spike'] = {}

        # Extract indices corresponding to the entries in the transmitted
        # buffers that must be copied into the input port map data arrays; these
        # are needed to support fan-out:
        self._in_port_dict_buf_ids = {}
        self._in_port_dict_buf_ids['gpot'] = {}
        self._in_port_dict_buf_ids['spike'] = {}

        # Lengths of input buffers:
        self._in_buf_len = {}
        self._in_buf_len['gpot'] = {}
        self._in_buf_len['spike'] = {}

        self._in_ids = self.routing_table.src_ids(self.id)
        self._in_ranks = [self.rank_to_id.inv[i] for i in self._in_ids]
        for in_id in self._in_ids:
            self.log_info('extracting input ports for %s' % in_id)

            # Get interfaces of pattern connecting the current module to
            # source module `in_id`; `int_1` is connected to the current
            # module, `int_0` is connected to the other module:
            pat = self.routing_table[in_id, self.id]['pattern']
            int_0 = self.routing_table[in_id, self.id]['int_0']
            int_1 = self.routing_table[in_id, self.id]['int_1']

            # Get ports in interface (`int_1`) connected to the current
            # module that are connected to the other module via the pattern:
            self._in_port_dict_ids['gpot'][in_id] = \
                self.pm['gpot'].ports_to_inds(pat.dest_idx(int_0, int_1, 'gpot', 'gpot'))
            self._in_port_dict_ids['spike'][in_id] = \
                self.pm['spike'].ports_to_inds(pat.dest_idx(int_0, int_1, 'spike', 'spike'))

			# Get the integer indices associated with the connected source ports
            # in the pattern interface connected to the source module `in_d`;
            # these are needed to copy received buffer contents into the current
            # module's port map data array:
            self._in_port_dict_buf_ids['gpot'][in_id] = \
                np.array(renumber_in_order(BasePortMapper(pat.gpot_ports(int_0).to_tuples()).
                        ports_to_inds(pat.src_idx(int_0, int_1, 'gpot', 'gpot', duplicates=True))))
            self._in_port_dict_buf_ids['spike'][in_id] = \
                np.array(renumber_in_order(BasePortMapper(pat.spike_ports(int_0).to_tuples()).
                        ports_to_inds(pat.src_idx(int_0, int_1, 'spike', 'spike', duplicates=True))))

            # The size of the input buffer to the current module must be the
            # same length as the output buffer of module `in_id`:
            self._in_buf_len['gpot'][in_id] = len(pat.src_idx(int_0, int_1, 'gpot', 'gpot'))
            self._in_buf_len['spike'][in_id] = len(pat.src_idx(int_0, int_1, 'spike', 'spike'))

    def _init_comm_bufs(self):
        """
        Buffers for sending/receiving data from other modules.

        Notes
        -----
        Must be executed after `_init_port_dicts()`.
        """

        # Buffers (and their interfaces and MPI types) for receiving data
        # transmitted from source modules:
        self._in_buf = {}
        self._in_buf['gpot'] = {}
        self._in_buf['spike'] = {}
        self._in_buf_int = {}
        self._in_buf_int['gpot'] = {}
        self._in_buf_int['spike'] = {}
        self._in_buf_mtype = {}
        self._in_buf_mtype['gpot'] = {}
        self._in_buf_mtype['spike'] = {}
        for in_id in self._in_ids:
            n_gpot = self._in_buf_len['gpot'][in_id]
            if n_gpot:
                self._in_buf['gpot'][in_id] = \
                    np.empty(n_gpot, self.pm['gpot'].dtype)
                self._in_buf_int['gpot'][in_id] = \
                    bufint(self._in_buf['gpot'][in_id])
                self._in_buf_mtype['gpot'][in_id] = \
                    dtype_to_mpi(self._in_buf['gpot'][in_id].dtype)
            else:
                self._in_buf['gpot'][in_id] = None

            n_spike = self._in_buf_len['spike'][in_id]
            if n_spike:
                self._in_buf['spike'][in_id] = \
                    np.empty(n_spike, self.pm['spike'].dtype)
                self._in_buf_int['spike'][in_id] = \
                    bufint(self._in_buf['spike'][in_id])
                self._in_buf_mtype['spike'][in_id] = \
                    dtype_to_mpi(self._in_buf['spike'][in_id].dtype)
            else:
                self._in_buf['spike'][in_id] = None

        # Buffers (and their interfaces and MPI types) for transmitting data to
        # destination modules:
        self._out_buf = {}
        self._out_buf['gpot'] = {}
        self._out_buf['spike'] = {}
        self._out_buf_int = {}
        self._out_buf_int['gpot'] = {}
        self._out_buf_int['spike'] = {}
        self._out_buf_mtype = {}
        self._out_buf_mtype['gpot'] = {}
        self._out_buf_mtype['spike'] = {}
        for out_id in self._out_ids:
            n_gpot = len(self._out_port_dict_ids['gpot'][out_id])
            if n_gpot:
                self._out_buf['gpot'][out_id] = \
                    np.empty(n_gpot, self.pm['gpot'].dtype)
                self._out_buf_int['gpot'][out_id] = \
                    bufint(self._out_buf['gpot'][out_id])
                self._out_buf_mtype['gpot'][out_id] = \
                    dtype_to_mpi(self._out_buf['gpot'][out_id].dtype)
            else:
                self._out_buf['gpot'][out_id] = None

            n_spike = len(self._out_port_dict_ids['spike'][out_id])
            if n_spike:
                self._out_buf['spike'][out_id] = \
                    np.empty(n_spike, self.pm['spike'].dtype)
                self._out_buf_int['spike'][out_id] = \
                    bufint(self._out_buf['spike'][out_id])
                self._out_buf_mtype['spike'][out_id] = \
                    dtype_to_mpi(self._out_buf['spike'][out_id].dtype)
            else:
                self._out_buf['spike'][out_id] = None

    def _sync(self):
        """
        Send output data and receive input data.
        """

        if self.time_sync:
            start = time.time()
        requests = []

        # For each destination module, extract elements from the current
        # module's port data array, copy them to a contiguous array, and
        # transmit the latter:
        for dest_id, dest_rank in zip(self._out_ids, self._out_ranks):

            # Copy data into destination buffer:
            if self._out_buf['gpot'][dest_id] is not None:
                self._out_buf['gpot'][dest_id][:] = \
                    self.data['gpot'][self._out_port_dict_ids['gpot'][dest_id]]
                if not self.time_sync:
                    self.log_info('gpot data sent to %s: %s' % \
                                  (dest_id, str(self._out_buf['gpot'][dest_id])))
                r = MPI.COMM_WORLD.Isend([self._out_buf_int['gpot'][dest_id],
                                          self._out_buf_mtype['gpot'][dest_id]],
                                         dest_rank, GPOT_TAG)
                requests.append(r)
            if self._out_buf['spike'][dest_id] is not None:
                self._out_buf['spike'][dest_id][:] = \
                    self.data['spike'][self._out_port_dict_ids['spike'][dest_id]]
                if not self.time_sync:
                    self.log_info('spike data sent to %s: %s' % \
                                  (dest_id, str(self._out_buf['spike'][dest_id])))
                r = MPI.COMM_WORLD.Isend([self._out_buf_int['spike'][dest_id],
                                          self._out_buf_mtype['spike'][dest_id]],
                                         dest_rank, SPIKE_TAG)
                requests.append(r)
            if not self.time_sync:
                self.log_info('sending to %s' % dest_id)
        if not self.time_sync:
            self.log_info('sent all data from %s' % self.id)

        # For each source module, receive elements and copy them into the
        # current module's port data array:
        for src_id, src_rank in zip(self._in_ids, self._in_ranks):
            if self._in_buf['gpot'][src_id] is not None:
                r = MPI.COMM_WORLD.Irecv([self._in_buf_int['gpot'][src_id],
                                          self._in_buf_mtype['gpot'][src_id]],
                                         source=src_rank, tag=GPOT_TAG)
                requests.append(r)
            if self._in_buf['spike'][src_id] is not None:
                r = MPI.COMM_WORLD.Irecv([self._in_buf_int['spike'][src_id],
                                          self._in_buf_mtype['spike'][src_id]],
                                         source=src_rank, tag=SPIKE_TAG)
                requests.append(r)
            if not self.time_sync:
                self.log_info('receiving from %s' % src_id)
        if requests:
            self.req.Waitall(requests)
        if not self.time_sync:
            self.log_info('all data were received by %s' % self.id)

        # Copy received elements into the current module's data array:
        for src_id in self._in_ids:
            if self._in_buf['gpot'][src_id] is not None:
                if not self.time_sync:
                    self.log_info('gpot data received from %s: %s' % \
                                  (src_id, str(self._in_buf['gpot'][src_id])))
                self.data['gpot'][self._in_port_dict_ids['gpot'][src_id]] = \
                    self._in_buf['gpot'][src_id][self._in_port_dict_buf_ids['gpot'][src_id]]
            if self._in_buf['spike'][src_id] is not None:
                if not self.time_sync:
                    self.log_info('spike data received from %s: %s' % \
                                  (src_id, str(self._in_buf['spike'][src_id])))
                self.data['spike'][self._in_port_dict_ids['spike'][src_id]] = \
                    self._in_buf['spike'][src_id][self._in_port_dict_buf_ids['spike'][src_id]]

        # Save timing data:
        if self.time_sync:
            stop = time.time()
            n_gpot = 0
            n_spike = 0
            for src_id in self._in_ids:
                n_gpot += len(self._in_buf['gpot'][src_id])
                n_spike += len(self._in_buf['spike'][src_id])
            self.log_info('sent timing data to master')
            self.intercomm.isend(['sync_time',
                                  (self.rank, self.steps, start, stop,
                                   n_gpot*self.pm['gpot'].dtype.itemsize+\
                                   n_spike*self.pm['spike'].dtype.itemsize)],
                                 dest=0, tag=self._ctrl_tag)
        else:
            self.log_info('saved all data received by %s' % self.id)

    def pre_run(self):
        """
        Code to run before main loop.

        This method is invoked by the `run()` method before the main loop is
        started.
        """

        self.log_info('running code before body of worker %s' % self.rank)

        # Initialize _out_port_dict and _in_port_dict attributes:
        self._init_port_dicts()

        # Initialize transmission buffers:
        self._init_comm_bufs()

        # Start timing the main loop:
        if self.time_sync:
            self.intercomm.isend(['start_time', (self.rank, time.time())],
                                 dest=0, tag=self._ctrl_tag)
            self.log_info('sent start time to manager')

    def post_run(self):
        """
        Code to run after main loop.

        This method is invoked by the `run()` method after the main loop is
        started.
        """

        self.log_info('running code after body of worker %s' % self.rank)

        # Stop timing the main loop before shutting down the emulation:
        if self.time_sync:
            self.intercomm.isend(['stop_time', (self.rank, time.time())],
                                 dest=0, tag=self._ctrl_tag)

            self.log_info('sent stop time to manager')

        # Send acknowledgment message:
        self.intercomm.isend(['done', self.rank], 0, self._ctrl_tag)
        self.log_info('done message sent to manager')

    def run_step(self):
        """
        Module work method.

        This method should be implemented to do something interesting with new
        input port data in the module's `pm` attribute and update the attribute's
        output port data if necessary. It should not interact with any other
        class attributes.
        """

        self.log_info('running execution step')

    def run(self):
        """
        Body of process.
        """

        # Don't allow keyboard interruption of process:
        with IgnoreKeyboardInterrupt():

            # Activate execution loop:
            super(Module, self).run()

    def do_work(self):
        """
        Work method.

        This method is repeatedly executed by the Worker instance after the
        instance receives a 'start' control message and until it receives a 'stop'
        control message.
        """

        # If the debug flag is set, don't catch exceptions so that
        # errors will lead to visible failures:
        if self.debug:

            # Run the processing step:
            self.run_step()

            # Synchronize:
            self._sync()
        else:

            # Run the processing step:
            catch_exception(self.run_step, self.log_info)

            # Synchronize:
            catch_exception(self._sync, self.log_info)

class Manager(mpi.WorkerManager):
    """
    Module manager.

    Instantiates, connects, starts, and stops modules comprised by an
    emulation. All modules and connections must be added to a module manager
    instance before they can be run.

    Attributes
    ----------
    ctrl_tag : int
        MPI tag to identify control messages.
    modules : dict
        Module instances. Keyed by module object ID.
    routing_table : routing_table.RoutingTable
        Table of data transmission connections between modules.
    rank_to_id : bidict.bidict
        Mapping between MPI ranks and module object IDs.
    """

    def __init__(self, required_args=['sel', 'sel_in', 'sel_out',
                                      'sel_gpot', 'sel_spike'],
                 ctrl_tag=CTRL_TAG):
        super(Manager, self).__init__(ctrl_tag)

        # Required constructor args:
        self.required_args = required_args

        # One-to-one mapping between MPI rank and module ID:
        self.rank_to_id = bidict.bidict()

        # Unique object ID:
        self.id = uid()

        # Set up a dynamic table to contain the routing table:
        self.routing_table = RoutingTable()

        # Number of emulation steps to run:
        self.steps = np.inf

        # Variables for timing run loop:
        self.start_time = 0.0
        self.stop_time = 0.0

        # Variables for computing throughput:
        self.counter = 0
        self.total_sync_time = 0.0
        self.total_sync_nbytes = 0.0
        self.received_data = {}

        # Average step synchronization time:
        self._average_step_sync_time = 0.0

        # Computed throughput (only updated after an emulation run):
        self._average_throughput = 0.0
        self._total_throughput = 0.0
        self.log_info('manager instantiated')

    @property
    def average_step_sync_time(self):
        """
        Average step synchronization time.
        """

        return self._average_step_sync_time
    @average_step_sync_time.setter
    def average_step_sync_time(self, t):
        self._average_step_sync_time = t

    @property
    def total_throughput(self):
        """
        Total received data throughput.
        """

        return self._total_throughput
    @total_throughput.setter
    def total_throughput(self, t):
        self._total_throughput = t

    @property
    def average_throughput(self):
        """
        Average received data throughput per step.
        """

        return self._average_throughput
    @average_throughput.setter
    def average_throughput(self, t):
        self._average_throughput = t

    def validate_args(self, target):
        """
        Check whether a class' constructor has specific arguments.

        Parameters
        ----------
        target : Module
            Module class to instantiate and run.

        Returns
        -------
        result : bool
            True if all of the required arguments are present, False otherwise.
        """

        arg_names = set(mpi.getargnames(target.__init__))
        for required_arg in self.required_args:
            if required_arg not in arg_names:
                return False
        return True

    def add(self, target, id, *args, **kwargs):
        """
        Add a module class to the emulation.

        Parameters
        ----------
        target : Module
            Module class to instantiate and run.
        id : str
            Identifier to use when connecting an instance of this class
            with an instance of some other class added to the emulation.
        args : sequence
            Sequential arguments to pass to the constructor of the class
            associated with identifier `id`.
        kwargs : dict
            Named arguments to pass to the constructor of the class
            associated with identifier `id`.
        """

        if not issubclass(target, Module):
            raise ValueError('target is not a Module subclass')

        # Selectors must be passed to the module upon instantiation;
        # the module manager must know about them to assess compatibility:
        # XXX: keep this commented out for the time being because it interferes
        # with instantiation of child classes (such as those in LPU.py):
        # if not self.validate_args(target):
        #    raise ValueError('class constructor missing required args')

        # Need to associate an ID with each module class
        # to instantiate; because the routing table's can potentially occupy
        # lots of space, we don't add it to the argument dict here - it is
        # broadcast to all processes separately and then added to the argument
        # dict in mpi_backend.py:
        kwargs['id'] = id
        kwargs['rank_to_id'] = self.rank_to_id
        rank = super(Manager, self).add(target, *args, **kwargs)
        self.rank_to_id[rank] = id

    def connect(self, id_0, id_1, pat, int_0=0, int_1=1):
        """
        Specify connection between two module instances with a Pattern instance.

        Parameters
        ----------
        id_0, id_1 : str
            Identifiers of module instances to connect.
        pat : Pattern
            Pattern instance.
        int_0, int_1 : int
            Which of the pattern's interfaces to connect to `id_0` and `id_1`,
            respectively.
        """

        if not isinstance(pat, Pattern):
            raise ValueError('pat is not a Pattern instance')
        if id_0 not in self.rank_to_id.values():
            raise ValueError('unrecognized module id %s' % id_0)
        if id_1 not in self.rank_to_id.values():
            raise ValueError('unrecognized module id %s' % id_1)
        if not (int_0 in pat.interface_ids and int_1 in pat.interface_ids):
            raise ValueError('unrecognized pattern interface identifiers')
        self.log_info('connecting modules {0} and {1}'
                      .format(id_0, id_1))

        # XXX Need to check for fan-in XXX

        # Store the pattern information in the routing table:
        self.log_info('updating routing table with pattern')
        if pat.is_connected(0, 1):
            self.routing_table[id_0, id_1] = {'pattern': pat,
                                              'int_0': int_0, 'int_1': int_1}
        if pat.is_connected(1, 0):
            self.routing_table[id_1, id_0] = {'pattern': pat,
                                              'int_0': int_1, 'int_1': int_0}

        self.log_info('connected modules {0} and {1}'.format(id_0, id_1))

    def process_worker_msg(self, msg):

        # Process timing data sent by workers:
        if msg[0] == 'start_time':
            rank, start_time = msg[1]
            self.log_info('start time data: %s' % str(msg[1]))
            if start_time < self.start_time or self.start_time == 0.0:
                self.start_time = start_time
                self.log_info('setting earliest start time: %s' % start_time)
        elif msg[0] == 'stop_time':
            rank, stop_time = msg[1]
            self.log_info('stop time data: %s' % str(msg[1]))
            if stop_time > self.stop_time or self.stop_time == 0.0:
                self.stop_time = stop_time
                self.log_info('setting latest stop time: %s' % stop_time)
        elif msg[0] == 'sync_time':
            rank, steps, start, stop, nbytes = msg[1]
            self.log_info('sync time data: %s' % str(msg[1]))

            # Collect timing data for each execution step:
            if steps not in self.received_data:
                self.received_data[steps] = {}
            self.received_data[steps][rank] = (start, stop, nbytes)

            # After adding the latest timing data for a specific step, check
            # whether data from all modules has arrived for that step:
            if set(self.received_data[steps].keys()) == set(self.rank_to_id.keys()):

                # Exclude the very first step to avoid including delays due to
                # PyCUDA kernel compilation:
                if steps != 0:

                    # The duration of an execution step is assumed to be the
                    # longest of the received intervals:
                    step_sync_time = max([(d[1]-d[0]) for d in self.received_data[steps].values()])

                    # Obtain the total number of bytes received by all of the
                    # modules during the execution step:
                    step_nbytes = sum([d[2] for d in self.received_data[steps].values()])

                    self.total_sync_time += step_sync_time
                    self.total_sync_nbytes += step_nbytes

                    self.average_throughput = (self.average_throughput*self.counter+\
                                              step_nbytes/step_sync_time)/(self.counter+1)
                    self.average_step_sync_time = (self.average_step_sync_time*self.counter+\
                                                   step_sync_time)/(self.counter+1)

                    self.counter += 1
                else:

                    # To exclude the time taken by the first step, set the start
                    # time to the latest stop time of the first step:
                    self.start_time = max([d[1] for d in self.received_data[steps].values()])
                    self.log_info('setting start time to skip first step: %s' % self.start_time)

                # Clear the data for the processed execution step so that
                # that the received_data dict doesn't consume unnecessary memory:
                del self.received_data[steps]

            # Compute throughput using accumulated timing data:
            if self.total_sync_time > 0:
                self.total_throughput = self.total_sync_nbytes/self.total_sync_time
            else:
                self.total_throughput = 0.0

    def wait(self):
        super(Manager, self).wait()
        self.log_info('avg step sync time/avg per-step throughput' \
                      '/total transm throughput/run loop duration:' \
                      '%s, %s, %s, %s' % \
                      (self.average_step_sync_time, self.average_throughput,
                       self.total_throughput, self.stop_time-self.start_time))

if __name__ == '__main__':
    import neurokernel.mpi_relaunch

    class MyModule(Module):
        """
        Example of derived module class.
        """

        def run_step(self):

            super(MyModule, self).run_step()

            # Do something with input graded potential data:
            self.log_info('input gpot port data: '+str(self.pm['gpot'][self.in_gpot_ports]))

            # Do something with input spike data:
            self.log_info('input spike port data: '+str(self.pm['spike'][self.in_spike_ports]))

            # Output random graded potential data:
            out_gpot_data = np.random.rand(len(self.out_gpot_ports))
            self.pm['gpot'][self.out_gpot_ports] = out_gpot_data
            self.log_info('output gpot port data: '+str(out_gpot_data))

            # Randomly select output ports to emit spikes:
            out_spike_data = np.random.randint(0, 2, len(self.out_spike_ports))
            self.pm['spike'][self.out_spike_ports] = out_spike_data
            self.log_info('output spike port data: '+str(out_spike_data))

    def make_sels(sel_in_gpot, sel_out_gpot, sel_in_spike, sel_out_spike):
        sel_in_gpot = Selector(sel_in_gpot)
        sel_out_gpot = Selector(sel_out_gpot)
        sel_in_spike = Selector(sel_in_spike)
        sel_out_spike = Selector(sel_out_spike)

        sel = sel_in_gpot+sel_out_gpot+sel_in_spike+sel_out_spike
        sel_in = sel_in_gpot+sel_in_spike
        sel_out = sel_out_gpot+sel_out_spike
        sel_gpot = sel_in_gpot+sel_out_gpot
        sel_spike = sel_in_spike+sel_out_spike

        return sel, sel_in, sel_out, sel_gpot, sel_spike

    logger = mpi.setup_logger(screen=True, file_name='neurokernel.log',
                              mpi_comm=MPI.COMM_WORLD, multiline=True)

    man = Manager()

    m1_sel_in_gpot = Selector('/a/in/gpot[0:2]')
    m1_sel_out_gpot = Selector('/a/out/gpot[0:2]')
    m1_sel_in_spike = Selector('/a/in/spike[0:2]')
    m1_sel_out_spike = Selector('/a/out/spike[0:2]')
    m1_sel, m1_sel_in, m1_sel_out, m1_sel_gpot, m1_sel_spike = \
        make_sels(m1_sel_in_gpot, m1_sel_out_gpot, m1_sel_in_spike, m1_sel_out_spike)
    N1_gpot = SelectorMethods.count_ports(m1_sel_gpot)
    N1_spike = SelectorMethods.count_ports(m1_sel_spike)

    m2_sel_in_gpot = Selector('/b/in/gpot[0:2]')
    m2_sel_out_gpot = Selector('/b/out/gpot[0:2]')
    m2_sel_in_spike = Selector('/b/in/spike[0:2]')
    m2_sel_out_spike = Selector('/b/out/spike[0:2]')
    m2_sel, m2_sel_in, m2_sel_out, m2_sel_gpot, m2_sel_spike = \
        make_sels(m2_sel_in_gpot, m2_sel_out_gpot, m2_sel_in_spike, m2_sel_out_spike)
    N2_gpot = SelectorMethods.count_ports(m2_sel_gpot)
    N2_spike = SelectorMethods.count_ports(m2_sel_spike)

    # Note that the module ID doesn't need to be listed in the specified
    # constructor arguments:
    m1_id = 'm1   '
    man.add(MyModule, m1_id, m1_sel, m1_sel_in, m1_sel_out,
            m1_sel_gpot, m1_sel_spike,
            np.zeros(N1_gpot, dtype=np.double),
            np.zeros(N1_spike, dtype=int),
            time_sync=True)
    m2_id = 'm2   '
    man.add(MyModule, m2_id, m2_sel, m2_sel_in, m2_sel_out,
            m2_sel_gpot, m2_sel_spike,
            np.zeros(N2_gpot, dtype=np.double),
            np.zeros(N2_spike, dtype=int),
            time_sync=True)

    # Make sure that all ports in the patterns' interfaces are set so
    # that they match those of the modules:
    pat12 = Pattern(m1_sel, m2_sel)
    pat12.interface[m1_sel_out_gpot] = [0, 'in', 'gpot']
    pat12.interface[m1_sel_in_gpot] = [0, 'out', 'gpot']
    pat12.interface[m1_sel_out_spike] = [0, 'in', 'spike']
    pat12.interface[m1_sel_in_spike] = [0, 'out', 'spike']
    pat12.interface[m2_sel_in_gpot] = [1, 'out', 'gpot']
    pat12.interface[m2_sel_out_gpot] = [1, 'in', 'gpot']
    pat12.interface[m2_sel_in_spike] = [1, 'out', 'spike']
    pat12.interface[m2_sel_out_spike] = [1, 'in', 'spike']
    pat12['/a/out/gpot[0]', '/b/in/gpot[0]'] = 1
    pat12['/a/out/gpot[1]', '/b/in/gpot[1]'] = 1
    pat12['/b/out/gpot[0]', '/a/in/gpot[0]'] = 1
    pat12['/b/out/gpot[1]', '/a/in/gpot[1]'] = 1
    pat12['/a/out/spike[0]', '/b/in/spike[0]'] = 1
    pat12['/a/out/spike[1]', '/b/in/spike[1]'] = 1
    pat12['/b/out/spike[0]', '/a/in/spike[0]'] = 1
    pat12['/b/out/spike[1]', '/a/in/spike[1]'] = 1

    check_compatible = True
    if check_compatible:
        m1_int = Interface.from_selectors(m1_sel, m1_sel_in, m1_sel_out,
                                          m1_sel_spike, m1_sel_gpot, m1_sel)
        m2_int = Interface.from_selectors(m2_sel, m2_sel_in, m2_sel_out,
                                          m2_sel_spike, m2_sel_gpot, m2_sel)
        assert m1_int.is_compatible(0, pat12.interface, 0, True)
        assert m2_int.is_compatible(0, pat12.interface, 1, True)
    man.connect(m1_id, m2_id, pat12, 0, 1)

    # Start emulation and allow it to run for a little while before shutting
    # down.  To set the emulation to exit after executing a fixed number of
    # steps, start it as follows and remove the sleep statement:
    # man.start(500)
    man.spawn()
    man.start(5)
    man.wait()
