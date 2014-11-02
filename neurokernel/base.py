#!/usr/bin/env python

"""
Base Neurokernel classes.
"""

import copy
import multiprocessing as mp
import os
import re
import string
import sys
import time
import collections

import bidict
from mpi4py import MPI
import numpy as np
import twiggy

import mpi
from ctx_managers import IgnoreKeyboardInterrupt, OnKeyboardInterrupt, \
     ExceptionOnSignal, TryExceptionOnSignal
from routing_table import RoutingTable
from uid import uid
from tools.misc import catch_exception
from pattern import Interface, Pattern
from plsel import PathLikeSelector, PortMapper

DATA_TAG = 0
CTRL_TAG = 1

class BaseModule(mpi.Worker):
    """
    Processing module.

    This class repeatedly executes a work method until it receives a
    quit message via its control network port. 

    Parameters
    ----------
    selector : str, unicode, or sequence
        Path-like selector describing the module's interface of 
        exposed ports.
    data : numpy.ndarray
        Data array to associate with ports. Array length must equal the number
        of ports in a module's interface.    
    columns : list of str
        Interface port attributes.
        Network port for controlling the module instance.
    id : str
        Module identifier. If no identifier is specified, a unique 
        identifier is automatically generated.
    routing_table : neurokernel.routing_table.RoutingTable
        Routing table describing data connections between modules. If no routing
        table is specified, the module will be executed in isolation.   
    debug : bool
        Debug flag. When True, exceptions raised during the work method
        are not be suppressed.

    Attributes
    ----------
    interface : Interface
        Object containing information about a module's ports.    
    pm : plsel.PortMapper
        Map between a module's ports and the contents of the `data` attribute.
    data : numpy.ndarray
        Array of data associated with a module's ports.
    """

    # Define properties to perform validation when the maximum number of
    # execution steps set:
    _steps = np.inf
    @property
    def steps(self):
        """
        Maximum number of steps to execute.
        """
        return self._steps
    @steps.setter
    def steps(self, value):
        if value <= 0:
            raise ValueError('invalid maximum number of steps')
        self.logger.info('maximum number of steps changed: %s -> %s' % (self._steps, value))
        self._steps = value

    def __init__(self, selector, data, columns=['interface', 'io', 'type'],
                 data_tag=DATA_TAG, ctrl_tag=CTRL_TAG,
                 id=None, routing_table=None, rank_to_id=None, 
                 debug=False):
        super(BaseModule, self).__init__(data_tag, ctrl_tag)
        self.debug = debug

        # Save routing table and mapping between MPI ranks and module IDs:
        self.routing_table = routing_table
        self.rank_to_id = rank_to_id

        # Generate a unique ID if none is specified:
        if id is None:
            self.id = uid()
        else:

            # Save routing table; if a unique ID was specified, it must be a node in
            # the routing table:
            if routing_table is not None and not routing_table.has_node(id):
                raise ValueError('routing table must contain specified module ID')
            self.id = id
                    
        # Logging:
        self.logger = twiggy.log.name(mpi.format_name('module %s' % self.id))

        # Create module interface given the specified ports:
        self.interface = Interface(selector, columns)

        # Set the interface ID to 0; we assume that a module only has one interface:
        self.interface[selector, 'interface'] = 0

        # Set up mapper between port identifiers and their associated data:
        assert len(data) == len(self.interface)
        self.data = data
        self.pm = PortMapper(self.data, selector)

    def _sync(self):
        """
        Send output data and receive input data.
        """

        req = MPI.Request()
        requests = []
        received = []
        idx_in_list = []

        # For each destination module, extract elements from the current
        # module's port data array, copy them to a contiguous array, and
        # transmit the latter:
        dest_ids = self.routing_table.dest_ids(self.id)
        for dest_id in dest_ids:            
            pat = self.routing_table[self.id, dest_id]['pattern']
            int_0 = self.routing_table[self.id, dest_id]['int_0']
            int_1 = self.routing_table[self.id, dest_id]['int_1']

            # Get source ports in current module that are connected to the
            # destination module:
            idx_out = pat.src_idx(int_0, int_1)
            data = self.pm[idx_out]
            dest_rank = self.rank_to_id[:dest_id]
            self.logger.info('data being sent to %s: %s' % (dest_id, str(data)))
            r = MPI.COMM_WORLD.Isend([data, MPI._typedict[data.dtype.char]],
                                     dest_rank)
            requests.append(r)
            self.logger.info('sending to %s' % dest_id)
        self.logger.info('sent all data from %s' % self.id)

        # For each source module, receive elements and copy them into the
        # current module's port data array:
        src_ids = self.routing_table.src_ids(self.id)
        for src_id in src_ids:
            pat = self.routing_table[src_id, self.id]['pattern']
            int_0 = self.routing_table[src_id, self.id]['int_0']
            int_1 = self.routing_table[src_id, self.id]['int_1']

            # Get destination ports in current module that are connected to the
            # source module:
            idx_in = pat.dest_idx(int_0, int_1)
            data = np.empty(np.shape(idx_in), self.pm.dtype)
            src_rank = self.rank_to_id[:src_id]
            r = MPI.COMM_WORLD.Irecv([data, MPI._typedict[data.dtype.char]],
                                     source=src_rank)
            requests.append(r)
            received.append(data)
            idx_in_list.append(idx_in)
            self.logger.info('receiving from %s' % src_id)
        req.Waitall(requests)
        self.logger.info('received all data received by %s' % self.id)

        # Copy received elements into the current module's data array:
        for data, idx_in in zip(received, idx_in_list):
            self.pm[idx_in] = data
        self.logger.info('saved all data received by %s' % self.id)

    def pre_run(self, *args, **kwargs):
        """
        Code to run before main module run loop.

        Code in this method will be executed after a module's process has been
        launched and all connectivity objects made available, but before the
        main run loop begins.
        """

        self.logger.info('performing pre-emulation operations')

    def post_run(self, *args, **kwargs):
        """
        Code to run after main module run loop.

        Code in this method will be executed after a module's main loop has
        terminated.
        """

        self.logger.info('performing post-emulation operations')

    def run_step(self):
        """
        Module work method.
    
        This method should be implemented to do something interesting with new 
        input port data in the module's `pm` attribute and update the attribute's
        output port data if necessary. It should not interact with any other 
        class attributes.
        """

        self.logger.info('running execution step')

    def run(self):
        """
        Body of process.
        """

        # Don't allow keyboard interruption of process:
        with IgnoreKeyboardInterrupt():

            # Perform any pre-emulation operations:
            self.pre_run()

            # Activate execution loop:
            super(BaseModule, self).run()

            # Perform any post-emulation operations:
            self.post_run()

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
            catch_exception(self.run_step, self.logger.info)

            # Synchronize:
            catch_exception(self._sync, self.logger.info)
        
class Manager(mpi.Manager):
    """
    Module manager.

    Instantiates, connects, starts, and stops modules comprised by an
    emulation. All modules and connections must be added to a module manager
    instance before they can be run.

    Attributes
    ----------
    data_tag : int
        MPI tag to identify data messages.
    ctrl_tag : int
        MPI tag to identify control messages.
    modules : dict
        Module instances. Keyed by module object ID.
    routing_table : routing_table.RoutingTable
        Table of data transmission connections between modules.
    rank_to_id : bidict.bidict
        Mapping between MPI ranks and module object IDs.
    """

    def __init__(self, mpiexec='mpiexec', mpiargs=(), data_tag=0, ctrl_tag=1):
        super(Manager, self).__init__(mpiexec, mpiargs, data_tag, ctrl_tag)

        # One-to-one mapping between MPI rank and module ID:
        self.rank_to_id = bidict.bidict()

        # Unique object ID:
        self.id = uid()
        
        # Set up a dynamic table to contain the routing table:
        self.routing_table = RoutingTable()

        # Number of emulation steps to run:
        self.steps = np.inf

        self.logger.info('manager instantiated')

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

        # if self._is_master():
        #     self.logger.info('in master! - skipping add')
        #     return

        assert issubclass(target, BaseModule)
        argnames = mpi.getargnames(target.__init__)

        # Selectors must be passed to the module upon instantiation;
        # the module manager must know about them to assess compatibility:
        assert 'selector' in argnames
        assert 'sel_in' in argnames
        assert 'sel_out' in argnames

        # Need to associate an ID and the routing table with each module class
        # to instantiate:
        kwargs['id'] = id
        kwargs['routing_table'] = self.routing_table
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

        Notes
        -----
        Assumes that the constructors of the module types contain a `selector`
        parameter.
        """

        assert isinstance(pat, Pattern)

        assert id_0 in self.rank_to_id.values()
        assert id_1 in self.rank_to_id.values()
        assert int_0 in pat.interface_ids and int_1 in pat.interface_ids

        # Check compatibility of the interfaces exposed by the modules and the
        # pattern; since the manager only contains module classes and not class
        # instances, we need to create Interface instances from the selectors
        # associated with the modules in order to test their compatibility:
        rank_0 = self.rank_to_id.inv[id_0]
        rank_1 = self.rank_to_id.inv[id_1]

        self.logger.info('checking compatibility of modules {0} and {1} and'
                         ' assigned pattern'.format(id_0, id_1))
        mod_int_0 = Interface(self._kwargs[rank_0]['selector'])
        mod_int_0[self._kwargs[rank_0]['selector']] = 0
        mod_int_1 = Interface(self._kwargs[rank_1]['selector'])
        mod_int_1[self._kwargs[rank_1]['selector']] = 0

        mod_int_0[self._kwargs[rank_0]['sel_in'], 'io'] = 'in'
        mod_int_0[self._kwargs[rank_0]['sel_out'], 'io'] = 'out'
        mod_int_1[self._kwargs[rank_1]['sel_in'], 'io'] = 'in'
        mod_int_1[self._kwargs[rank_1]['sel_out'], 'io'] = 'out'

        assert mod_int_0.is_compatible(0, pat.interface, int_0)
        assert mod_int_1.is_compatible(0, pat.interface, int_1)

        # XXX Need to check for fan-in XXX

        # Store the pattern information in the routing table:
        self.logger.info('updating routing table with pattern')
        if pat.is_connected(0, 1):
            self.routing_table[id_0, id_1] = {'pattern': pat, 
                                              'int_0': int_0, 'int_1': int_1}
        if pat.is_connected(1, 0):
            self.routing_table[id_1, id_0] = {'pattern': pat, 
                                              'int_0': int_1, 'int_1': int_0}
if __name__ == '__main__':
    class MyModule(BaseModule):
        """
        Example of derived module class.
        """

        def __init__(self, selector, sel_in, sel_out, data,
                     columns=['interface', 'io', 'type'],
                     data_tag=DATA_TAG, ctrl_tag=CTRL_TAG,
                     id=None, routing_table=None, rank_to_id=None, debug=False):
            super(MyModule, self).__init__(selector, data, columns, data_tag, ctrl_tag,
                                           id, routing_table, rank_to_id, debug)

            assert PathLikeSelector.is_in(sel_in, selector)
            assert PathLikeSelector.is_in(sel_out, selector)
            assert PathLikeSelector.are_disjoint(sel_in, sel_out)

            self.interface[sel_in, 'io'] = 'in'            
            self.interface[sel_out, 'io'] = 'out'

            # Find the input and output ports:
            self.in_ports = self.interface.in_ports().to_tuples()
            self.out_ports = self.interface.out_ports().to_tuples()

        def run_step(self):

            super(MyModule, self).run_step()

            # Do something with input data; for the sake of illustration, we
            # just record the current values:
            self.logger.info('input port data: '+str(self.pm[self.in_ports]))

            # Output random data:
            self.pm[self.out_ports] = np.random.rand(len(self.out_ports))
            self.logger.info('output port data: '+str(self.pm[self.out_ports]))

    logger = mpi.setup_logger(stdout=sys.stdout, file_name='log',
                              mpi_comm=MPI.COMM_WORLD, multiline=True)

    man = Manager()

    m1_int_sel = '/a[0:5]'; m1_int_sel_in = '/a[0:2]'; m1_int_sel_out = '/a[2:5]'
    m2_int_sel = '/b[0:5]'; m2_int_sel_in = '/b[0:3]'; m2_int_sel_out = '/b[3:5]'
    m3_int_sel = '/c[0:4]'; m3_int_sel_in = '/c[0:2]'; m3_int_sel_out = '/c[2:4]'

    # Note that the module ID doesn't need to be listed in the specified
    # constructor arguments:
    m1_id = 'm1   '
    man.add(MyModule, m1_id, m1_int_sel, m1_int_sel_in, m1_int_sel_out,
            np.zeros(5, dtype=np.float),
            ['interface', 'io', 'type'],
            DATA_TAG, CTRL_TAG)
    m2_id = 'm2   '
    man.add(MyModule, m2_id, m2_int_sel, m2_int_sel_in, m2_int_sel_out,
            np.zeros(5, dtype=np.float),
            ['interface', 'io', 'type'],
            DATA_TAG, CTRL_TAG)
    # m3_id = 'm3   '
    # man.add(MyModule, m3_id, m3_int_sel, m3_int_sel_in, m3_int_sel_out,
    #         np.zeros(4, dtype=np.float),
    #         ['interface', 'io', 'type'],
    #         DATA_TAG, CTRL_TAG)

    # Make sure that all ports in the patterns' interfaces are set so 
    # that they match those of the modules:
    pat12 = Pattern(m1_int_sel, m2_int_sel)
    pat12.interface[m1_int_sel_out, 'io'] = 'in'
    pat12.interface[m1_int_sel_in, 'io'] = 'out'
    pat12.interface[m2_int_sel_in, 'io'] = 'out'
    pat12.interface[m2_int_sel_out, 'io'] = 'in'
    pat12['/a[2]', '/b[0]'] = 1
    pat12['/a[3]', '/b[1]'] = 1
    pat12['/b[3]', '/a[0]'] = 1
    pat12['/b[4]', '/a[1]'] = 1
    man.connect(m1_id, m2_id, pat12, 0, 1)

    # pat23 = Pattern(m2_int_sel, m3_int_sel)
    # pat23.interface[m2_int_sel_out, 'io'] = 'in'
    # pat23.interface[m2_int_sel_in, 'io'] = 'out'
    # pat23.interface[m3_int_sel_in, 'io'] = 'out'
    # pat23.interface[m3_int_sel_out, 'io'] = 'in'
    # pat23['/b[4]', '/c[0]'] = 1
    # pat23['/c[2]', '/b[2]'] = 1
    # man.connect(m2_id, m3_id, pat23, 0, 1)

    # pat31 = Pattern(m3_int_sel, m1_int_sel)
    # pat31.interface[m3_int_sel_out, 'io'] = 'in'
    # pat31.interface[m1_int_sel_in, 'io'] = 'out'
    # pat31.interface[m3_int_sel_in, 'io'] = 'out'
    # pat31.interface[m1_int_sel_out, 'io'] = 'in'
    # pat31['/c[3]', '/a[1]'] = 1
    # pat31['/a[4]', '/c[1]'] = 1
    # man.connect(m3_id, m1_id, pat31, 0, 1)

    # Start emulation and allow it to run for a little while before shutting
    # down.  To set the emulation to exit after executing a fixed number of
    # steps, start it as follows and remove the sleep statement:
    # man.start(steps=500)

    man.run()
    man.start()
#    man.steps(10)
    man.start()
    time.sleep(3)
    man.stop()
    man.quit()
