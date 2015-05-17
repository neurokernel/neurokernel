#!/usr/bin/env python

import atexit
import time

import bidict
from mpi4py import MPI
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.tools import dtype_to_ctype
import twiggy

from base_gpu import BaseModule, CTRL_TAG
import base_gpu
from mixins import LoggerMixin
import mpi
from tools.mpi import MPIOutput
from tools.gpu import bufint, set_by_inds
from tools.logging import setup_logger
from tools.misc import catch_exception, dtype_to_mpi
from uid import uid
from pattern import Interface, Pattern
from plsel import SelectorMethods
from pm_gpu import GPUPortMapper

# MPI tags for distinguishing messages associated with different port types:
GPOT_TAG = CTRL_TAG+1
SPIKE_TAG = CTRL_TAG+2

class Module(BaseModule):
    def __init__(self, sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns=['interface', 'io', 'type'],
                 ctrl_tag=CTRL_TAG, gpot_tag=GPOT_TAG, spike_tag=SPIKE_TAG,
                 id=None, device=None,
                 routing_table=None, rank_to_id=None,
                 debug=False, time_sync=False):

        # Call super for BaseModule rather than Module because most of the
        # functionality of the former's constructor must be overridden in any case:
        super(BaseModule, self).__init__(ctrl_tag)
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

        # Initialize GPU here so as to be able to initialize a port mapper
        # containing GPU memory:
        self._init_gpu()

        # This is needed to ensure that MPI_Finalize is called before PyCUDA
        # attempts to clean up; see
        # https://groups.google.com/forum/#!topic/mpi4py/by0Rd5q0Ayw
        atexit.register(MPI.Finalize)

        # Manually register the file close method associated with MPIOutput
        # so that it is called by atexit before MPI.Finalize() (if the file is
        # closed after MPI.Finalize() is called, an error will occur):
        for k, v in twiggy.emitters.iteritems():
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
        if not SelectorMethods.is_in(sel_in, sel):
            raise ValueError('input port selector not in selector of all ports')
        if not SelectorMethods.is_in(sel_out, sel):
            raise ValueError('output port selector not in selector of all ports')
        if not SelectorMethods.are_disjoint(sel_in, sel_out):
            raise ValueError('input and output port selectors not disjoint')

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
        self.data['gpot'] = gpuarray.to_gpu(data_gpot)
        self.data['spike'] = gpuarray.to_gpu(data_spike)
        self.pm = {}
        self.pm['gpot'] = GPUPortMapper(sel_gpot, self.data['gpot'], make_copy=False)
        self.pm['spike'] = GPUPortMapper(sel_spike, self.data['spike'], make_copy=False)

    def _init_port_dicts(self):
        """
        Initial dictionaries of source/destination ports in current module.
        """

        # Extract identifiers of source ports in the current module's interface
        # for all modules receiving output from the current module:
        self._out_port_dict = {}
        self._out_port_dict['gpot'] = {}
        self._out_port_dict['spike'] = {}
        self._out_port_dict_ids = {}
        self._out_port_dict_ids['gpot'] = {}
        self._out_port_dict_ids['spike'] = {}

        self._out_ids = self.routing_table.dest_ids(self.id)
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
            self._out_port_dict['gpot'][out_id] = \
                    pat.src_idx(int_0, int_1, 'gpot', 'gpot')
            self._out_port_dict_ids['gpot'][out_id] = \
                    self.pm['gpot'].ports_to_inds(self._out_port_dict['gpot'][out_id])
            self._out_port_dict['spike'][out_id] = \
                    pat.src_idx(int_0, int_1, 'spike', 'spike')
            self._out_port_dict_ids['spike'][out_id] = \
                    self.pm['spike'].ports_to_inds(self._out_port_dict['spike'][out_id])
           
        # Extract identifiers of destination ports in the current module's
        # interface for all modules sending input to the current module:
        self._in_port_dict = {}
        self._in_port_dict['gpot'] = {}
        self._in_port_dict['spike'] = {}
        self._in_port_dict_ids = {}
        self._in_port_dict_ids['gpot'] = {}
        self._in_port_dict_ids['spike'] = {}

        self._in_ids = self.routing_table.src_ids(self.id)
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
            self._in_port_dict['gpot'][in_id] = \
                    pat.dest_idx(int_0, int_1, 'gpot', 'gpot')
            self._in_port_dict_ids['gpot'][in_id] = \
                    self.pm['gpot'].ports_to_inds(self._in_port_dict['gpot'][in_id])
            self._in_port_dict['spike'][in_id] = \
                    pat.dest_idx(int_0, int_1, 'spike', 'spike')
            self._in_port_dict_ids['spike'][in_id] = \
                    self.pm['spike'].ports_to_inds(self._in_port_dict['spike'][in_id])

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
            self._in_buf['gpot'][in_id] = \
                gpuarray.empty(len(self._in_port_dict_ids['gpot'][in_id]),
                               self.pm['gpot'].dtype)
            self._in_buf_int['gpot'][in_id] = bufint(self._in_buf['gpot'][in_id])
            self._in_buf_mtype['gpot'][in_id] = \
                dtype_to_mpi(self._in_buf['gpot'][in_id].dtype)
            self._in_buf['spike'][in_id] = \
                gpuarray.empty(len(self._in_port_dict_ids['spike'][in_id]),
                               self.pm['spike'].dtype)
            self._in_buf_int['spike'][in_id] = bufint(self._in_buf['spike'][in_id])
            self._in_buf_mtype['spike'][in_id] = \
                dtype_to_mpi(self._in_buf['spike'][in_id].dtype)

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
            self._out_buf['gpot'][out_id] = \
                gpuarray.empty(len(self._out_port_dict_ids['gpot'][out_id]),
                               self.pm['gpot'].dtype)
            self._out_buf_int['gpot'][out_id] = bufint(self._out_buf['gpot'][out_id])
            self._out_buf_mtype['gpot'][out_id] = \
                dtype_to_mpi(self._out_buf['gpot'][out_id].dtype)

            self._out_buf['spike'][out_id] = \
                gpuarray.empty(len(self._out_port_dict_ids['spike'][out_id]),
                               self.pm['spike'].dtype)
            self._out_buf_int['spike'][out_id] = bufint(self._out_buf['spike'][out_id])
            self._out_buf_mtype['spike'][out_id] = \
                dtype_to_mpi(self._out_buf['spike'][out_id].dtype)

    def _sync(self):
        """
        Send output data and receive input data.
        """

        if self.time_sync:
            start = time.time()
        req = MPI.Request()
        requests = []

        # For each destination module, extract elements from the current
        # module's port data array, copy them to a contiguous array, and
        # transmit the latter:
        dest_ids = self.routing_table.dest_ids(self.id)
        for dest_id in dest_ids:            
            
            # Copy data into destination buffer:
            dest_rank = self.rank_to_id[:dest_id]
            set_by_inds(self._out_buf['gpot'][dest_id],
                        self._out_port_dict_ids['gpot'][dest_id],
                        self.data['gpot'], 'src')
            set_by_inds(self._out_buf['spike'][dest_id],
                        self._out_port_dict_ids['spike'][dest_id],
                        self.data['spike'], 'src')

            if not self.time_sync:
                self.log_info('gpot data being sent to %s: %s' % \
                              (dest_id, str(self._out_buf['gpot'][dest_id])))
                self.log_info('spike data being sent to %s: %s' % \
                              (dest_id, str(self._out_buf['spike'][dest_id])))
            r = MPI.COMM_WORLD.Isend([self._out_buf_int['gpot'][dest_id],
                                      self._out_buf_mtype['gpot'][dest_id]],
                                     dest_rank, GPOT_TAG)
            requests.append(r)
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
        src_ids = self.routing_table.src_ids(self.id)
        for src_id in src_ids:
            src_rank = self.rank_to_id[:src_id]
            r = MPI.COMM_WORLD.Irecv([self._in_buf_int['gpot'][src_id],
                                      self._in_buf_mtype['gpot'][src_id]],
                                     source=src_rank, tag=GPOT_TAG)
            requests.append(r)
            r = MPI.COMM_WORLD.Irecv([self._in_buf_int['spike'][src_id],
                                      self._in_buf_mtype['spike'][src_id]],
                                     source=src_rank, tag=SPIKE_TAG)
            requests.append(r)
            if not self.time_sync:
                self.log_info('receiving from %s' % src_id)
        req.Waitall(requests)
        if not self.time_sync:
            self.log_info('received all data received by %s' % self.id)

        # Copy received elements into the current module's data array:
        n_gpot = 0
        n_spike = 0
        for src_id in src_ids:
            ind_in_gpot = self._in_port_dict_ids['gpot'][src_id]
            self.pm['gpot'].set_by_inds(ind_in_gpot, self._in_buf['gpot'][src_id])
            n_gpot += len(self._in_buf['gpot'][src_id])
            ind_in_spike = self._in_port_dict_ids['spike'][src_id]
            self.pm['spike'].set_by_inds(ind_in_spike, self._in_buf['spike'][src_id])
            n_spike += len(self._in_buf['spike'][src_id])

        # Save timing data:
        if self.time_sync:
            stop = time.time()
            self.log_info('sent timing data to master')
            self.intercomm.isend(['sync_time',
                                  (self.rank, self.steps, start, stop,
                                   n_gpot*self.pm['gpot'].dtype.itemsize+\
                                   n_spike*self.pm['spike'].dtype.itemsize)],
                                 dest=0, tag=self._ctrl_tag)
        else:
            self.log_info('saved all data received by %s' % self.id)

class Manager(base_gpu.Manager):
    def __init__(self, required_args=['sel', 'sel_in', 'sel_out',
                                      'sel_gpot', 'sel_spike'],
                 ctrl_tag=CTRL_TAG):
        super(Manager, self).__init__(required_args, ctrl_tag)
 
    def add(self, target, id, *args, **kwargs):
        if not issubclass(target, Module):
            raise ValueError('target is not a Module subclass')
        super(Manager, self).add(target, id, *args, **kwargs)

    def connect(self, id_0, id_1, pat, int_0=0, int_1=1, compat_check=True):
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

        # Check compatibility of the interfaces exposed by the modules and the
        # pattern; since the manager only contains module classes and not class
        # instances, we need to create Interface instances from the selectors
        # associated with the modules in order to test their compatibility:
        if compat_check:
            rank_0 = self.rank_to_id.inv[id_0]
            rank_1 = self.rank_to_id.inv[id_1]

            self.log_info('checking compatibility of modules {0} and {1} and'
                             ' assigned pattern'.format(id_0, id_1))
            mod_int_0 = Interface(self._kwargs[rank_0]['sel'])
            mod_int_0[self._kwargs[rank_0]['sel']] = 0
            mod_int_1 = Interface(self._kwargs[rank_1]['sel'])
            mod_int_1[self._kwargs[rank_1]['sel']] = 0

            mod_int_0[self._kwargs[rank_0]['sel_in'], 'io'] = 'in'
            mod_int_0[self._kwargs[rank_0]['sel_out'], 'io'] = 'out'
            mod_int_0[self._kwargs[rank_0]['sel_gpot'], 'type'] = 'gpot'
            mod_int_0[self._kwargs[rank_0]['sel_spike'], 'type'] = 'spike'
            mod_int_1[self._kwargs[rank_1]['sel_in'], 'io'] = 'in'
            mod_int_1[self._kwargs[rank_1]['sel_out'], 'io'] = 'out'
            mod_int_1[self._kwargs[rank_1]['sel_gpot'], 'type'] = 'gpot'
            mod_int_1[self._kwargs[rank_1]['sel_spike'], 'type'] = 'spike'

            if not mod_int_0.is_compatible(0, pat.interface, int_0, True):
                raise ValueError('module %s interface incompatible '
                                 'with pattern interface %s' % (id_0, int_0))
            if not mod_int_1.is_compatible(0, pat.interface, int_1, True):
                raise ValueError('module %s interface incompatible '
                                 'with pattern interface %s' % (id_1, int_1))

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
            out_gpot_data = gpuarray.to_gpu(np.random.rand(len(self.out_gpot_ports)))
            self.pm['gpot'][self.out_gpot_ports] = out_gpot_data
            self.log_info('output gpot port data: '+str(out_gpot_data))

            # Randomly select output ports to emit spikes:
            out_spike_data = gpuarray.to_gpu(np.random.randint(0, 2, len(self.out_spike_ports)))
            self.pm['spike'][self.out_spike_ports] = out_spike_data
            self.log_info('output spike port data: '+str(out_spike_data))

    logger = mpi.setup_logger(screen=True, file_name='neurokernel.log',
                              mpi_comm=MPI.COMM_WORLD, multiline=True)

    man = Manager()

    m1_int_sel_in_gpot = '/a/in/gpot0,/a/in/gpot1'
    m1_int_sel_out_gpot = '/a/out/gpot0,/a/out/gpot1'
    m1_int_sel_in_spike = '/a/in/spike0,/a/in/spike1'
    m1_int_sel_out_spike = '/a/out/spike0,/a/out/spike1'
    m1_int_sel = ','.join([m1_int_sel_in_gpot, m1_int_sel_out_gpot,
                           m1_int_sel_in_spike, m1_int_sel_out_spike])
    m1_int_sel_in = ','.join((m1_int_sel_in_gpot, m1_int_sel_in_spike))
    m1_int_sel_out = ','.join((m1_int_sel_out_gpot, m1_int_sel_out_spike))
    m1_int_sel_gpot = ','.join((m1_int_sel_in_gpot, m1_int_sel_out_gpot))
    m1_int_sel_spike = ','.join((m1_int_sel_in_spike, m1_int_sel_out_spike))
    N1_gpot = SelectorMethods.count_ports(m1_int_sel_gpot)
    N1_spike = SelectorMethods.count_ports(m1_int_sel_spike)

    m2_int_sel_in_gpot = '/b/in/gpot0,/b/in/gpot1'
    m2_int_sel_out_gpot = '/b/out/gpot0,/b/out/gpot1'
    m2_int_sel_in_spike = '/b/in/spike0,/b/in/spike1'
    m2_int_sel_out_spike = '/b/out/spike0,/b/out/spike1'
    m2_int_sel = ','.join([m2_int_sel_in_gpot, m2_int_sel_out_gpot,
                           m2_int_sel_in_spike, m2_int_sel_out_spike])
    m2_int_sel_in = ','.join((m2_int_sel_in_gpot, m2_int_sel_in_spike))
    m2_int_sel_out = ','.join((m2_int_sel_out_gpot, m2_int_sel_out_spike))
    m2_int_sel_gpot = ','.join((m2_int_sel_in_gpot, m2_int_sel_out_gpot))
    m2_int_sel_spike = ','.join((m2_int_sel_in_spike, m2_int_sel_out_spike))
    N2_gpot = SelectorMethods.count_ports(m2_int_sel_gpot)
    N2_spike = SelectorMethods.count_ports(m2_int_sel_spike)

    # Note that the module ID doesn't need to be listed in the specified
    # constructor arguments:
    m1_id = 'm1   '
    man.add(MyModule, m1_id, m1_int_sel, m1_int_sel_in, m1_int_sel_out,
            m1_int_sel_gpot, m1_int_sel_spike,
            np.zeros(N1_gpot, dtype=np.double),
            np.zeros(N1_spike, dtype=int),
            ['interface', 'io', 'type'],
            CTRL_TAG, GPOT_TAG, SPIKE_TAG, device=0, time_sync=True)
    m2_id = 'm2   '
    man.add(MyModule, m2_id, m2_int_sel, m2_int_sel_in, m2_int_sel_out,
            m2_int_sel_gpot, m2_int_sel_spike,
            np.zeros(N2_gpot, dtype=np.double),
            np.zeros(N2_spike, dtype=int),
            ['interface', 'io', 'type'],
            CTRL_TAG, GPOT_TAG, SPIKE_TAG, device=1, time_sync=True)

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
    man.connect(m1_id, m2_id, pat12, 0, 1)

    # Start emulation and allow it to run for a little while before shutting
    # down.  To set the emulation to exit after executing a fixed number of
    # steps, start it as follows and remove the sleep statement:
    # man.start(500)
    man.spawn()
    man.start(5)
    man.wait()
