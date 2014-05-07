#!/usr/bin/env python

"""
Base Neurokernel classes.
"""

from contextlib import contextmanager
import copy
import multiprocessing as mp
import os
import re
import string
import sys
import threading
import time
import collections

import bidict
import numpy as np
import scipy.sparse
import scipy as sp
import twiggy
import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
import msgpack_numpy as msgpack

from ctrl_proc import ControlledProcess, LINGER_TIME
from ctx_managers import IgnoreKeyboardInterrupt, OnKeyboardInterrupt, \
     ExceptionOnSignal, TryExceptionOnSignal
from tools.comm import is_poll_in, get_random_port
from routing_table import RoutingTable
from uid import uid
from tools.misc import catch_exception

PORT_DATA = 5000
PORT_CTRL = 5001

class BaseModule(ControlledProcess):
    """
    Processing module.

    This class repeatedly executes a work method until it receives a
    quit message via its control port.

    Parameters
    ----------
    port_data : int
        Port to use when communicating with broker.
    port_ctrl : int
        Port used by broker to control module.
    id : str
        Module identifier. If no identifier is specified, a unique identifier is
        automatically generated.

    Attributes
    ----------
    conn_dict : dict of BaseConnectivity
        Connectivity objects connecting the module instance with
        other module instances.
    in_ids : list of int
        List of source module IDs.
    out_ids : list of int
        List of destination module IDs.

    Methods
    -------
    run()
        Body of process.
    run_step(data)
        Processes the specified data and returns a result for
        transmission to other modules.

    Notes
    -----
    If the ports specified upon instantiation are None, the module
    instance ignores the network entirely.

    Children of the BaseModule class should also contain attributes containing
    the connectivity objects.

    """

    # Define properties to perform validation when connectivity status
    # is set:
    _net = 'none'
    @property
    def net(self):
        """
        Network connectivity.
        """
        return self._net
    @net.setter
    def net(self, value):
        if value not in ['none', 'ctrl', 'in', 'out', 'full']:
            raise ValueError('invalid network connectivity value')
        self.logger.info('net status changed: %s -> %s' % (self._net, value))
        self._net = value

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

    def __init__(self, port_data=PORT_DATA, port_ctrl=PORT_CTRL, id=None):

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

        # Initial connectivity:
        self.net = 'none'

        # List used for storing outgoing data; each
        # entry is a tuple whose first entry is the source or destination
        # module ID and whose second entry is the data:
        self._out_data = []

        # Objects describing connectivity between this module and other modules
        # keyed by the IDs of the other modules:
        self._conn_dict = {}

        # Dictionary containing ports of destination modules that receive input
        # from this module; must be initialized immediately before an emulation
        # begins running:
        self._out_idx_dict = {}

    @property
    def N(self):
        """
        Number of ports exposed by module.

        Notes
        -----
        Must be overwritten to return the actual number of ports.
        """

        raise NotImplementedError('N must be implemented')

    @property
    def all_ids(self):
        """
        IDs of modules to which the current module is connected.
        """

        return [c.other_mod(self.id) for c in self._conn_dict.values()]

    @property
    def in_ids(self):
        """
        IDs of modules that send data to this module.
        """

        return [c.other_mod(self.id) for c in self._conn_dict.values() if \
                c.is_connected(c.other_mod(self.id), self.id)]

    @property
    def out_ids(self):
        """
        IDs of modules that receive data from this module.
        """

        return [c.other_mod(self.id) for c in self._conn_dict.values() if \
                c.is_connected(self.id, c.other_mod(self.id))]

    def add_conn(self, conn):
        """
        Add the specified connectivity object.

        Parameters
        ----------
        conn : BaseConnectivity
            Connectivity object.

        Notes
        -----
        The module's ID must be one of the two IDs specified in the
        connnectivity object.
        """

        if not isinstance(conn, BaseConnectivity):
            raise ValueError('invalid connectivity object')
        if self.id not in [conn.A_id, conn.B_id]:
            raise ValueError('connectivity object must contain module ID')
        self.logger.info('connecting to %s' % conn.other_mod(self.id))

        # The connectivity instances associated with this module are keyed by
        # the ID of the other module:
        self._conn_dict[conn.other_mod(self.id)] = conn

        # Update internal connectivity based upon contents of connectivity
        # object. When the add_conn() method is invoked, the module's internal
        # connectivity is always upgraded to at least 'ctrl':
        if self.net == 'none':
            self.net = 'ctrl'
        if conn.is_connected(self.id, conn.other_mod(self.id)):
            old_net = self.net
            if self.net == 'ctrl':
                self.net = 'out'
            elif self.net == 'in':
                self.net = 'full'
            self.logger.info('net status changed: %s -> %s' % (old_net, self.net))
        if conn.is_connected(conn.other_mod(self.id), self.id):
            old_net = self.net
            if self.net == 'ctrl':
                self.net = 'in'
            elif self.net == 'out':
                self.net = 'full'
            self.logger.info('net status changed: %s -> %s' % (old_net, self.net))

    def _ctrl_stream_shutdown(self):
        """
        Shut down control port handler's stream and ioloop.
        """

        try:
            self.stream_ctrl.flush()
            self.stream_ctrl.stop_on_recv()
            self.ioloop_ctrl.stop()
        except IOError:
            self.logger.info('streams already closed')
        except:
            self.logger.info('other error occurred')
        else:
            self.logger.info('ctrl stream shut down')

    def _ctrl_handler(self, msg):
        """
        Control port handler.
        """

        self.logger.info('recv ctrl message: %s' % str(msg))
        if msg[0] == 'quit':
            self._ctrl_stream_shutdown()

            # Force the module's main loop to exit:
            self.running = False
            ack = 'shutdown'

        # One can define additional messages to be recognized by the control
        # handler:        
        # elif msg[0] == 'conn':
        #     self.logger.info('conn payload: '+str(msgpack.unpackb(msg[1])))
        #     ack = 'ack'
        else:
            ack = 'ack'

        self.sock_ctrl.send(ack)
        self.logger.info('sent to manager: %s' % ack)

    def _init_net(self):
        """
        Initialize network connection.
        """

        # Initialize control port handler:
        self.logger.info('initializing ctrl network connection')
        super(BaseModule, self)._init_net()

        if self.net == 'none':
            self.logger.info('not initializing data network connection')
        else:

            # Don't allow interrupts to prevent the handler from
            # completely executing each time it is called:
            with IgnoreKeyboardInterrupt():
                self.logger.info('initializing data network connection')

                # Use a nonblocking port for the data interface; set
                # the linger period to prevent hanging on unsent
                # messages when shutting down:
                self.sock_data = self.zmq_ctx.socket(zmq.DEALER)
                self.sock_data.setsockopt(zmq.IDENTITY, self.id)
                self.sock_data.setsockopt(zmq.LINGER, LINGER_TIME)
                self.sock_data.connect("tcp://localhost:%i" % self.port_data)
                self.logger.info('network connection initialized')

                # Set up a poller for detecting incoming data:
                self.data_poller = zmq.Poller()
                self.data_poller.register(self.sock_data, zmq.POLLIN)

    def _get_in_data(self, in_dict):
        """
        Get input data from incoming transmission buffer.

        Input data received from other modules is used to populate the specified
        data structures.

        Parameters
        ----------
        in_dict : dict of numpy.ndarray of float
            Dictionary of data from other modules keyed by source module ID.
        """

        self.logger.info('retrieving input')
        for in_id in self.in_ids:
            if in_id in self._in_data.keys() and self._in_data[in_id]:
                in_dict[in_id] = self._in_data[in_id].popleft()

    def _put_out_data(self, out):
        """
        Put output data in outgoing transmission buffer.

        Using the indices of the ports in destination modules that receive input
        from this module instance, data extracted from the module's neurons is
        staged for output transmission.

        Parameter
        ---------
        out : numpy.ndarray of float
            Output data.
        """

        self.logger.info('populating output buffer')

        # Clear output buffer before populating it:
        self._out_data = []

        # Use indices of destination ports to select which values need to be
        # transmitted to each destination module:
        for out_id in self.out_ids:
            self._out_data.append((out_id, np.asarray(out)[self._out_idx_dict[out_id]]))

    def _sync(self):
        """
        Send output data and receive input data.

        Notes
        -----
        Assumes that the attributes used for input and output already
        exist.

        Each message is a tuple containing a module ID and data; for
        outbound messages, the ID is that of the destination module.
        for inbound messages, the ID is that of the source module.
        Data is serialized before being sent and unserialized when
        received.
        """

        if self.net in ['none', 'ctrl']:
            self.logger.info('not synchronizing with network')
        else:
            self.logger.info('synchronizing with network')

            # Send outbound data:
            if self.net in ['out', 'full']:

                # Send all data in outbound buffer:
                send_ids = self.out_ids
                for out_id, data in self._out_data:
                    self.sock_data.send(msgpack.packb((out_id, data)))
                    send_ids.remove(out_id)
                    self.logger.info('sent to   %s: %s' % (out_id, str(data)))

                # Send data tuples containing None to those modules for which no
                # actual data was generated to satisfy the barrier condition:
                for out_id in send_ids:
                    self.sock_data.send(msgpack.packb((out_id, None)))
                    self.logger.info('sent to   %s: %s' % (out_id, None))

                # All output IDs should be sent data by this point:
                self.logger.info('sent data to all output IDs')

            # Receive inbound data:
            if self.net in ['in', 'full']:
                # Wait until inbound data is received from all source modules:  
                while not all((q for q in self._in_data.itervalues())):
                    # Use poller to avoid blocking:
                    if is_poll_in(self.sock_data, self.data_poller):
                        in_id, data = msgpack.unpackb(self.sock_data.recv())
                        self.logger.info('recv from %s: %s ' % (in_id, str(data)))

                        # Ignore incoming data containing None:
                        if data is not None:
                            self._in_data[in_id].append(data)

                    # Stop the synchronization if a quit message has been received:
                    if not self.running:
                        self.logger.info('run loop stopped - stopping sync')
                        break
                self.logger.info('recv data from all input IDs')

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

    def run_step(self, in_dict, out):
        """
        Perform a single step of computation.

        This method should be implemented to do something interesting with its
        arguments. It should not interact with any other class attributes.
        """

        self.logger.info('running execution step')

    def run(self):
        """
        Body of process.
        """

        # Don't allow keyboard interruption of process:
        self.logger.info('starting')
        with IgnoreKeyboardInterrupt():

            # Initialize environment:
            self._init_net()

            # Initialize Buffer for incoming data.
            # Dict used to store the incoming data keyed by the source module id.
            # Each value is a queue buferring the received data
            self._in_data = {k:collections.deque() for k in self.in_ids}

            # Extract indices of source ports for all modules receiving output
            # once so that they don't need to be repeatedly extracted during the
            # emulation:
            self._out_idx_dict = \
              {out_id:self._conn_dict[out_id].src_idx(self.id, out_id) for \
               out_id in self.out_ids}

            # Perform any pre-emulation operations:
            self.pre_run()

            self.running = True
            curr_steps = 0
            while curr_steps < self._steps:
                self.logger.info('execution step: %s' % curr_steps)

                # Clear data structures for passing data to and from the
                # run_step method:
                in_dict = {}
                out = []

                # Get input data:
                catch_exception(self._get_in_data,self.logger.info,in_dict)

                # Run the processing step:
                catch_exception(self.run_step,self.logger.info,in_dict, out)

                # Prepare the generated data for output:
                catch_exception(self._put_out_data,self.logger.info,out)

                # Synchronize:
                catch_exception(self._sync,self.logger.info)

                # Exit run loop when a quit message has been received:
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

class Broker(ControlledProcess):
    """
    Broker for communicating between modules.

    Waits to receive data from all input modules before transmitting the
    collected data to destination modules.

    Parameters
    ----------
    port_data : int
        Port to use for communication with modules.
    port_ctrl : int
        Port used to control modules.

    Methods
    -------
    run()
        Body of process.
    sync()
        Synchronize with network.
    """

    def __init__(self, port_data=PORT_DATA, port_ctrl=PORT_CTRL,
                 routing_table=None):
        super(Broker, self).__init__(port_ctrl, uid())

        # Logging:
        self.logger = twiggy.log.name('broker %s' % self.id)

        # Data port:
        if port_data == port_ctrl:
            raise ValueError('data and control ports must differ')
        self.port_data = port_data

        # Routing table:
        self.routing_table = routing_table

        # Buffers used to accumulate data to route:
        self.data_to_route = []

        # A dictionary keyed by routing table coords.
        # Each value represents the difference between number of data
        # messages received for that routing relation and the current number
        # of executed steps.

    def _ctrl_handler(self, msg):
        """
        Control port handler.
        """

        self.logger.info('recv: '+str(msg))
        if msg[0] == 'quit':
            try:
                self.stream_ctrl.flush()
                self.stream_data.flush()
                self.stream_ctrl.stop_on_recv()
                self.stream_data.stop_on_recv()
                self.ioloop.stop()
            except IOError:
                self.logger.info('streams already closed')
            except Exception as e:
                self.logger.info('other error occurred: '+e.message)
            self.sock_ctrl.send('ack')
            self.logger.info('sent to  broker: ack')

    def _data_handler(self, msg):
        """
        Data port handler.

        Notes
        -----
        Assumes that each message contains a source module ID
        (provided by zmq) and a serialized tuple; the tuple contains
        the destination module ID and the data to be transmitted.
        """

        if len(msg) != 2:
            self.logger.info('skipping malformed message: %s' % str(msg))
        else:

            # When a message arrives, increase the corresponding received_count
            in_id = msg[0]
            out_id, data = msgpack.unpackb(msg[1])
            self.logger.info('recv from %s: %s' % (in_id, data))
            # Increase the appropriate count in recv_counts by 1
            self.recv_counts[(in_id,out_id)] += 1
            self.data_to_route.append((in_id, out_id, data))
            # When data with source/destination IDs corresponding to
            # every entry in the routing table has been received upto
            # current time step, deliver the data in the buffer:
            if all((c for c in self.recv_counts.values())):
                self.logger.info('recv from all modules')
                for in_id, out_id, data in self.data_to_route:
                    self.logger.info('sent to   %s: %s' % (out_id, data))

                    # Route to the destination ID and send the source ID
                    # along with the data:
                    self.sock_data.send_multipart([out_id,
                                                   msgpack.packb((in_id, data))])

                # Reset the incoming data buffer
                self.data_to_route = []
                # Decrease all values in recv_counts to indicate that an
                # execution time_step has been succesfully completed
                for k in self.recv_counts.iterkeys(): self.recv_counts[k]-=1
                self.logger.info('----------------------')

    def _init_ctrl_handler(self):
        """
        Initialize control port handler.
        """

        # Set the linger period to prevent hanging on unsent messages
        # when shutting down:
        self.logger.info('initializing ctrl handler')
        self.sock_ctrl = self.zmq_ctx.socket(zmq.DEALER)
        self.sock_ctrl.setsockopt(zmq.IDENTITY, self.id)
        self.sock_ctrl.setsockopt(zmq.LINGER, LINGER_TIME)
        self.sock_ctrl.connect('tcp://localhost:%i' % self.port_ctrl)

        self.stream_ctrl = ZMQStream(self.sock_ctrl, self.ioloop)
        self.stream_ctrl.on_recv(self._ctrl_handler)

    def _init_data_handler(self):
        """
        Initialize data port handler.
        """

        # Set the linger period to prevent hanging on unsent
        # messages when shutting down:
        self.logger.info('initializing data handler')
        self.sock_data = self.zmq_ctx.socket(zmq.ROUTER)
        self.sock_data.setsockopt(zmq.LINGER, LINGER_TIME)
        self.sock_data.bind("tcp://*:%i" % self.port_data)

        self.stream_data = ZMQStream(self.sock_data, self.ioloop)
        self.stream_data.on_recv(self._data_handler)

    def _init_net(self):
        """
        Initialize the network connection.
        """

        # Since the broker must behave like a reactor, the event loop
        # is started in the main thread:
        self.zmq_ctx = zmq.Context()
        self.ioloop = IOLoop.instance()
        self._init_ctrl_handler()
        self._init_data_handler()
        self.ioloop.start()

    def run(self):
        """
        Body of process.
        """

        # Don't allow keyboard interruption of process:
        self.logger.info('starting')
        with IgnoreKeyboardInterrupt():
            self.recv_counts = dict(zip(self.routing_table.coords,\
                        np.zeros(len(self.routing_table.coords),dtype=np.int32)))
            self._init_net()
        self.logger.info('exiting')

class BaseConnectivity(object):
    """
    Intermodule connectivity.

    Stores the connectivity between two LPUs as a series of sparse matrices.
    Every entry in an instance of the class has the following indices:

    - source module ID
    - source port ID
    - destination module ID
    - destination port ID
    - connection number (when two ports are connected by more than one connection)
    - parameter name (the default is 'conn' for simple connectivity)
 
    Each connection may therefore have several parameters; parameters associated
    with nonexistent connections (i.e., those whose 'conn' parameter is set to
    0) should be ignored.
    
    Parameters
    ----------
    N_A : int
        Number of ports to interface with on module A.
    N_B: int
        Number of ports to interface with on module B.
    N_mult: int
        Maximum supported number of connections between any two neurons
        (default 1). Can be raised after instantiation.
    A_id : str
        First module ID (default 'A').
    B_id : str
        Second module ID (default 'B').

    Attributes
    ----------
    nbytes : int
        Approximate number of bytes occupied by object.
    
    Methods
    -------
    N(id)
        Number of ports associated with the specified module.
    is_connected(src_id, dest_id)
        Returns True of at least one connection
        exists between `src_id` and `dest_id`.
    other_mod(id)
        Returns the ID of the other module connected by the object to
        the one specified as `id`.
    dest_idx(src_id, dest_id, src_ports)
        Indices of ports in module `dest_id` with incoming 
        connections from module `src_id`.
    dest_mask(src_id, dest_id, src_ports)
        Mask of ports in module `dest_id` with incoming
        connections from module `src_id`.
    src_idx(src_id, dest_id, dest_ports)
        Indices of ports in module `src_id` with outgoing
        connections to module `dest_id`.
    src_mask(src_id, dest_id, dest_ports)
        Mask of ports in module `src_id` with outgoing
        connections to module `dest_id`.
    transpose()
        Returns a BaseConnectivity instance with the source and destination
        flipped.
    
    Examples
    --------
    The first connection between port 0 in LPU A with port 3 in LPU B can
    be accessed as c['A',0,'B',3,0]. The 'weight' parameter associated with this
    connection can be accessed as c['A',0,'B',3,0,'weight']
    
    Notes
    -----
    Since connections between LPUs should necessarily not contain any recurrent
    connections, it is more efficient to store the inter-LPU connections in two
    separate matrices that respectively map to and from the ports in each LPU
    rather than a large matrix whose dimensions comprise the total number of
    ports in both LPUs. Matrices that describe connections between A and B
    have dimensions (N_A, N_B), while matrices that describe connections between
    B and A have dimensions (N_B, N_A).
    
    """

    def __init__(self, N_A, N_B, N_mult=1, A_id='A', B_id='B'):

        # Unique object ID:
        self.id = uid()

        # The number of ports in both of the LPUs must be nonzero:
        assert N_A != 0
        assert N_B != 0

        # The maximum number of connections between any two ports must be
        # nonzero:
        assert N_mult != 0

        # The module IDs must be non-null and nonidentical:
        assert A_id != B_id
        assert len(A_id) != 0
        assert len(B_id) != 0
        
        self.N_A = N_A
        self.N_B = N_B
        self.N_mult = N_mult
        self.A_id = A_id
        self.B_id = B_id

        # Strings indicating direction between modules connected by instances of
        # the class:
        self._AtoB = '/'.join((A_id, B_id))
        self._BtoA = '/'.join((B_id, A_id))
        
        # All matrices are stored in this dict:
        self._data = {}

        # Keys corresponding to each connectivity direction are stored in the
        # following lists:
        self._keys_by_dir = {self._AtoB: [],
                             self._BtoA: []}

        # Create connectivity matrices for both directions; the key structure
        # is source module/dest module/connection #/parameter name. Note that
        # the matrices associated with A -> B have the dimensions (N_A, N_B)
        # while those associated with B -> have the dimensions (N_B, N_A):
        key = self._make_key(self._AtoB, 0, 'conn')
        self._data[key] = self._make_matrix((self.N_A, self.N_B), int)
        self._keys_by_dir[self._AtoB].append(key)        
        key = self._make_key(self._BtoA, 0, 'conn')
        self._data[key] = self._make_matrix((self.N_B, self.N_A), int)
        self._keys_by_dir[self._BtoA].append(key)

    def _validate_mod_names(self, A_id, B_id):
        """
        Raise an exception if the specified module names are not recognized.
        """
        
        if set((A_id, B_id)) != set((self.A_id, self.B_id)):
            raise ValueError('invalid module ID')
        
    def N(self, id):
        """
        Return number of ports associated with the specified module.
        """
        
        if id == self.A_id:
            return self.N_A
        elif id == self.B_id:
            return self.N_B
        else:
            raise ValueError('invalid module ID')

    def other_mod(self, id):
        """
        Given the specified module ID, return the ID to which the object
        connects it.
        """

        if id == self.A_id:
            return self.B_id
        elif id == self.B_id:
            return self.A_id
        else:
            raise ValueError('invalid module ID')

    def is_connected(self, src_id, dest_id):
        """
        Returns true if there is at least one connection from
        the specified source module to the specified destination module.        
        """

        self._validate_mod_names(src_id, dest_id)
        for k in self._keys_by_dir['/'.join((src_id, dest_id))]:
            if self._data[k].nnz:
                return True
        return False
    
    def src_mask(self, src_id='', dest_id='', dest_ports=slice(None, None)):
        """
        Mask of source ports with connections to destination ports.

        Parameters
        ----------
        src_id, dest_id : str
           Module IDs. If no IDs are specified, the IDs stored in
           attributes `A_id` and `B_id` are used in that order.
        dest_ports : int or slice
           Only look for source ports with connections to the specified
           destination ports.

        Examples
        --------
        >>> c = BaseConnectivity(3, 2)
        >>> c['A', 1, 'B', 0] = 1
        >>> all(c.src_mask() == [False, True, False])
        True
        >>> all(c.src_mask(dest_ports=1) == [False, False, False])
        True
        
        """

        if src_id == '' and dest_id == '':
            src_id = self.A_id
            dest_id = self.B_id

        self._validate_mod_names(src_id, dest_id)
        dir = '/'.join((src_id, dest_id))
            
        # XXX It isn't necessary to consider all of the connectivity matrices if
        # multapses are assumed to always have an entry in the first
        # connectivity matrix:
        all_dest_idx = np.arange(self.N(dest_id))[dest_ports]
        result = np.zeros(self.N(src_id), dtype=bool)
        for k in self._keys_by_dir[dir]:

            # Only look at the 'conn' parameter:
            if k.endswith('/conn'):
                result[:] = result+ \
                    [np.asarray([bool(np.intersect1d(all_dest_idx, r).size) \
                                     for r in self._data[k].rows])]
        return result

    def src_idx(self, src_id='', dest_id='', dest_ports=slice(None, None)):
        """
        Indices of source ports with connections to destination ports.

        Examples
        --------
        >>> c = BaseConnectivity(3, 2)
        >>> c['A', 1, 'B', 0] = 1
        >>> all(c.src_idx() == [1])
        True
        >>> all(c.src_idx(dest_ports=1) == [])
        True
        
        See Also
        --------
        BaseConnectivity.src_mask        
        """

        if src_id == '' and dest_id == '':
            src_id = self.A_id
            dest_id = self.B_id
        mask = self.src_mask(src_id, dest_id, dest_ports)
        return np.arange(self.N(src_id))[mask]
    
    def dest_mask(self, src_id='', dest_id='', src_ports=slice(None, None)):
        """
        Mask of destination ports with connections to source ports.

        Parameters
        ----------
        src_id, dest_id : str
           Module IDs. If no IDs are specified, the IDs stored in
           attributes `A_id` and `B_id` are used in that order.
        src_ports : int or slice
           Only look for destination ports with connections to the specified
           source ports.

        Examples
        --------
        >>> c = BaseConnectivity(3, 2)
        >>> c['A', 1, 'B', 0] = 1
        >>> all(c.dest_mask() == [True, False])
        True
        >>> all(c.dest_mask(src_ports=0) == [False, False])
        True
           
        """

        if src_id == '' and dest_id == '':
            src_id = self.A_id
            dest_id = self.B_id

        self._validate_mod_names(src_id, dest_id)
        dir = '/'.join((src_id, dest_id))
            
        # XXX It isn't necessary to consider all of the connectivity matrices if
        # multapses are assumed to always have an entry in the first
        # connectivity matrix:
        result = np.zeros(self.N(dest_id), dtype=bool)
        for k in self._keys_by_dir[dir]:

            # Only look at the 'conn' parameter:
            if k.endswith('/conn'):
                for r in self._data[k].rows[src_ports]:
                    result[r] = True
        return result
    
    def dest_idx(self, src_id='', dest_id='', src_ports=slice(None, None)):
        """
        Indices of destination ports with connections to source ports.

        Examples
        --------
        >>> c = BaseConnectivity(3, 2)
        >>> c['A', 1, 'B', 0] = 1
        >>> all(c.dest_idx() == [0])
        True
        >>> all(c.dest_idx(src_ports=0) == [])
        True
        
        See Also
        --------
        BaseConnectivity.dest_mask        
        """

        if src_id == '' and dest_id == '':
            src_id = self.A_id
            dest_id = self.B_id
        mask = self.dest_mask(src_id, dest_id, src_ports)
        return np.arange(self.N(dest_id))[mask]
    
    @property
    def nbytes(self):
        """
        Approximate number of bytes required by the class instance.

        Notes
        -----
        Only accounts for nonzero values in sparse matrices.
        """

        count = 0
        for key in self._data.keys():
            count += self._data[key].dtype.itemsize*self._data[key].nnz
        return count

    def _indent_str(self, s, indent=0):
        """
        Indent a string by the specified number of spaces.

        Parameters
        ----------
        s : str
            String to indent.
        indent : int
            Number of spaces to prepend to each line in the string.
        """
        
        return re.sub('^(.*)', indent*' '+r'\1', s, flags=re.MULTILINE)
    
    def _format_array(self, a, indent=0):
        """
        Format an array for printing.

        Parameters
        ----------
        a : 2D array_like
            Array to format.
        indent : int
            Number of columns by which to indent the formatted array.
        
        Returns
        -------
        result : str
            Formatted array.            
        """

        if scipy.sparse.issparse(a):            
            return self._indent_str(a.toarray().__str__(), indent)
        else:
            return self._indent_str(np.asarray(a).__str__(), indent)
        
    def __repr__(self):
        result = '%s -> %s\n' % (self.A_id, self.B_id)
        result += '-----------\n'
        for key in self._keys_by_dir[self._AtoB]:
            result += key + '\n'
            result += self._format_array(self._data[key]) + '\n'
        result += '\n%s -> %s\n' % (self.B_id, self.A_id)
        result += '-----------\n'
        for key in self._keys_by_dir[self._BtoA]:
            result += key + '\n'
            result += self._format_array(self._data[key]) + '\n'
        return result
        
    def _make_key(self, *args):
        """
        Create a unique key for a matrix of connection properties.
        """
        
        return string.join(map(str, args), '/')

    def _make_matrix(self, shape, dtype=np.double):
        """
        Create a sparse matrix of the specified shape.
        """

        # scipy.sparse doesn't support sparse arrays of strings;
        # we therefore use an ordinary ndarray of objects:
        if np.issubdtype(dtype, str):
            return np.empty(shape, dtype=np.object)
        else:
            return sp.sparse.lil_matrix(shape, dtype=dtype)

    def multapses(self, src_id, src_idx, dest_id, dest_idx):
        """
        Return number of multapses for the specified connection.
        """

        self._validate_mod_names(src_id, dest_id)
        dir = '/'.join((src_id, dest_id))
        count = 0
        for k in self._keys_by_dir[dir]:
            conn, name = k.split('/')[2:]
            conn = int(conn)
            if name == 'conn' and \
                self.get(src_id, src_idx, dest_id, dest_idx, conn, name):
                count += 1
        return count

    def _get_sparse(self, src_id, src_idx, dest_id, dest_idx, conn, param):
        """
        Retrieve a value or values in the connectivity class instance and return
        as scalar or sparse.
        """

        if src_id == '' and dest_id == '':
            dir = self._AtoB
        else:
            self._validate_mod_names(src_id, dest_id)
        dir = '/'.join((src_id, dest_id))
        assert type(conn) == int
        
        return self._data[self._make_key(dir, conn, param)][src_idx, dest_idx]

    def get(self, src_id, src_idx, dest_id, dest_idx, conn=0, param='conn'):
        """
        Retrieve a value in the connectivity class instance.
        """

        result = self._get_sparse(src_id, src_idx, dest_id, dest_idx, conn, param)
        if scipy.sparse.issparse(result):
            return result.toarray()
        else:
            return result

    def set(self, src_id, src_idx, dest_id, dest_idx, conn=0, param='conn', val=1):
        """
        Set a value in the connectivity class instance.

        Notes
        -----
        Creates a new storage matrix when the one specified doesn't exist.        
        """

        if src_id == '' and dest_id == '':
            dir = self._AtoB
        else:
            self._validate_mod_names(src_id, dest_id)
        dir = '/'.join((src_id, dest_id))
        assert type(conn) == int
        
        key = self._make_key(dir, conn, param)
        if not self._data.has_key(key):

            # XX should ensure that inserting a new matrix for an existing param
            # uses the same type as the existing matrices for that param XX
            if dir == self._AtoB:
                self._data[key] = \
                    self._make_matrix((self.N_A, self.N_B), type(val))
            else:
                self._data[key] = \
                    self._make_matrix((self.N_B, self.N_A), type(val))
            self._keys_by_dir[dir].append(key)

            # Increment the maximum number of connections between two ports as
            # needed:
            if conn+1 > self.N_mult:
                self.N_mult += 1
                
        self._data[key][src_idx, dest_idx] = val

    def transpose(self):
        """
        Returns an object instance with the source and destination LPUs flipped.
        """

        c = BaseConnectivity(self.N_B, self.N_A, self.N_mult,
                             A_id=self.B_id, B_id=self.A_id)
        c._keys_by_dir[self._AtoB] = []
        c._keys_by_dir[self._BtoA] = []
        for old_key in self._data.keys():

            # Reverse the direction in the key:
            key_split = old_key.split('/')
            A_id, B_id = key_split[0:2]
            new_dir = '/'.join((B_id, A_id))
            new_key = '/'.join([new_dir]+key_split[2:])
            c._data[new_key] = self._data[old_key].T           
            c._keys_by_dir[new_dir].append(new_key)
        return c

    @property
    def T(self):
        return self.transpose()
    
    def __getitem__(self, s):        
        return self.get(*s)

    def __setitem__(self, s, val):
        self.set(*s, val=val)
        
class BaseManager(object):
    """
    Module manager.

    Parameters
    ----------
    port_data : int
        Port to use for communication with modules.
    port_ctrl : int
        Port used to control modules.

    """

    def __init__(self, port_data=PORT_DATA, port_ctrl=PORT_CTRL):

        # Unique object ID:
        self.id = uid()

        self.logger = twiggy.log.name('manage %s' % self.id)
        self.port_data = port_data
        self.port_ctrl = port_ctrl

        # Set up a router socket to communicate with other topology
        # components; linger period is set to 0 to prevent hanging on
        # unsent messages when shutting down:
        self.zmq_ctx = zmq.Context()
        self.sock_ctrl = self.zmq_ctx.socket(zmq.ROUTER)
        self.sock_ctrl.setsockopt(zmq.LINGER, LINGER_TIME)
        self.sock_ctrl.bind("tcp://*:%i" % self.port_ctrl)

        # Set up a poller for detecting acknowledgements to control messages:
        self.ctrl_poller = zmq.Poller()
        self.ctrl_poller.register(self.sock_ctrl, zmq.POLLIN)
        
        # Data structures for storing broker, module, and connectivity instances:
        self.brok_dict = bidict.bidict()
        self.mod_dict = bidict.bidict()
        self.conn_dict = bidict.bidict()

        # Set up a dynamic table to contain the routing table:
        self.routing_table = RoutingTable()

        # Number of emulation steps to run:
        self.steps = np.inf

    def connect(self, m_A, m_B, conn):
        """
        Connect two module instances with a connectivity object instance.

        Parameters
        ----------
        m_A, m_B : BaseModule
           Module instances to connect
        conn : BaseConnectivity
           Connectivity object instance.
                
        """

        if not isinstance(m_A, BaseModule) or \
            not isinstance(m_B, BaseModule) or \
            not isinstance(conn, BaseConnectivity):
            raise ValueError('invalid type')

        if m_A.id not in [conn.A_id, conn.B_id] or \
            m_B.id not in [conn.A_id, conn.B_id]:
            raise ValueError('connectivity object doesn\'t contain modules\' IDs')

        if not((m_A.N == conn.N_A and m_B.N == conn.N_B) or \
               (m_A.N == conn.N_B and m_B.N == conn.N_A)):
            raise ValueError('modules and connectivity objects are incompatible')

        # Add the module and connection instances to the internal
        # dictionaries of the manager instance if they are not already there:
        if m_A.id not in self.mod_dict:
            self.add_mod(m_A)
        if m_B.id not in self.mod_dict:
            self.add_mod(m_B)
        if conn.id not in self.conn_dict:
            self.add_conn(conn)

        # Connect the modules with the specified connectivity module:
        m_A.add_conn(conn)
        m_B.add_conn(conn)

        # Update the routing table:
        if conn.is_connected(m_A.id, m_B.id):
            self.routing_table[m_A.id, m_B.id] = 1
        if conn.is_connected(m_B.id, m_A.id):
            self.routing_table[m_B.id, m_A.id] = 1

    @property
    def N_brok(self):
        """
        Number of brokers.
        """
        return len(self.brok_dict)

    @property
    def N_mod(self):
        """
        Number of modules.
        """
        return len(self.mod_dict)

    @property
    def N_conn(self):
        """
        Number of connectivity objects.
        """

        return len(self.conn_dict)

    def add_brok(self, b=None):
        """
        Add or create a broker instance to the emulation.
        """

        # TEMPORARY: only allow one broker:
        if self.N_brok == 1:
            raise RuntimeError('only one broker allowed')

        if not isinstance(b, Broker):
            b = Broker(port_data=self.port_data,
                       port_ctrl=self.port_ctrl, routing_table=self.routing_table)
        self.brok_dict[b.id] = b
        self.logger.info('added broker %s' % b.id)
        return b

    def add_mod(self, m=None):
        """
        Add or create a module instance to the emulation.
        """

        if not isinstance(m, BaseModule):
            m = BaseModule(port_data=self.port_data, port_ctrl=self.port_ctrl)
        self.mod_dict[m.id] = m
        self.logger.info('added module %s' % m.id)
        return m

    def add_conn(self, c):
        """
        Add a connectivity instance to the emulation.
        """

        if not isinstance(c, BaseConnectivity):
            raise ValueError('invalid connectivity object')
        self.conn_dict[c.id] = c
        self.logger.info('added connectivity %s' % c.id)
        return c

    def start(self, steps=np.inf):
        """
        Start execution of all processes.

        Parameters
        ----------
        steps : int
            Maximum number of steps to execute.
        """

        self.steps = steps
        with IgnoreKeyboardInterrupt():
            for b in self.brok_dict.values():
                b.start()
            for m in self.mod_dict.values():
                m.steps = steps
                m.start()

    def send_ctrl_msg(self, i, *msg):
        """
        Send control message(s) to a module.
        """

        self.sock_ctrl.send_multipart([i]+msg)
        self.logger.info('sent to   %s: %s' % (i, msg))
        while True:
            if is_poll_in(self.sock_ctrl, self.ctrl_poller):
                j, data = self.sock_ctrl.recv_multipart()
                self.logger.info('recv from %s: ack' % j)
                break

    def stop_brokers(self):
        """
        Stop execution of all brokers.
        """

        self.logger.info('stopping all brokers')
        for i in self.brok_dict.keys():
            self.logger.info('sent to   %s: quit' % i)
            self.sock_ctrl.send_multipart([i, 'quit'])
            self.brok_dict[i].join(1)
        self.logger.info('all brokers stopped')
        
    def join_modules(self, send_quit=False):
        """
        Wait until all modules have stopped.

        Parameters
        ----------
        send_quit : bool
            If True, send quit messages to all modules.            
        """

        self.logger.info('waiting for modules to shut down')
        recv_ids = self.mod_dict.keys()
        while recv_ids:
            i = recv_ids[0]
            
            # Send quit messages to all live modules:
            if send_quit:                
                self.logger.info(str(recv_ids))
                self.logger.info('sent to   %s: quit' % i)
                self.sock_ctrl.send_multipart([i, 'quit'])

            # If a module acknowledges receiving a quit message,
            # wait for it to shutdown:
            if is_poll_in(self.sock_ctrl, self.ctrl_poller):
                 j, data = self.sock_ctrl.recv_multipart()
                 self.logger.info('recv from %s: %s' % (j, data))                 
                 if j in recv_ids and data == 'shutdown':
                     self.logger.info('waiting for module %s to shut down' % j)
                     recv_ids.remove(j)
                     self.mod_dict[j].join(1)
                     self.logger.info('module %s shut down' % j)
                     
            # Sometimes quit messages are received but the acknowledgements are
            # lost; if so, the module will eventually shutdown:
            # XXX this shouldn't be necessary XXX
            if not self.mod_dict[i].is_alive() and i in recv_ids:
                self.logger.info('%s shutdown without ack' % i)
                recv_ids.remove(i)                
        self.logger.info('all modules stopped')

    def stop(self):
        """
        Stop execution of an emulation.
        """

        if np.isinf(self.steps):
            self.logger.info('stopping all modules')
            send_quit = True
        else:
            send_quit = False
        self.join_modules(send_quit)
        self.stop_brokers()
        
def setup_logger(file_name='neurokernel.log', screen=True, port=None):
    """
    Convenience function for setting up logging with twiggy.

    Parameters
    ----------
    file_name : str
        Log file.
    screen : bool
        If true, write logging output to stdout.
    port : int
        If set to a ZeroMQ port number, publish 
        logging output to that port.

    Returns
    -------
    logger : twiggy.logger.Logger
        Logger object.

    Bug
    ---
    To use the ZeroMQ output class, it must be added as an emitter within each
    process.
    """

    if file_name:
        file_output = \
          twiggy.outputs.FileOutput(file_name, twiggy.formats.line_format, 'w')
        twiggy.addEmitters(('file', twiggy.levels.DEBUG, None, file_output))

    if screen:
        screen_output = \
          twiggy.outputs.StreamOutput(twiggy.formats.line_format,
                                      stream=sys.stdout)
        twiggy.addEmitters(('screen', twiggy.levels.DEBUG, None, screen_output))

    if port:
        port_output = ZMQOutput('tcp://*:%i' % port,
                               twiggy.formats.line_format)
        twiggy.addEmitters(('port', twiggy.levels.DEBUG, None, port_output))

    return twiggy.log.name(('{name:%s}' % 12).format(name='main'))

if __name__ == '__main__':
    from neurokernel.tools.misc import rand_bin_matrix

    np.random.seed(0)

    class MyModule(BaseModule):
        """
        Example of derived module class.
        """

        def __init__(self, N, id, port_data=PORT_DATA, port_ctrl=PORT_CTRL):                     
            super(MyModule, self).__init__(port_data, port_ctrl)
            self.data = np.zeros(N, np.float64)
            
        @property
        def N(self):
            return len(self.data)
        
        def run_step(self, in_dict, out):
            super(MyModule, self).run_step(in_dict, out)

            out[:] = np.random.rand(self.N)

        def run(self):
            super(MyModule, self).run()
            
    # Set up logging:
    logger = setup_logger()

    # Set up emulation:
    man = BaseManager(get_random_port(), get_random_port())
    man.add_brok()

    m1 = man.add_mod(MyModule(2, 'm1   ', man.port_data, man.port_ctrl))
    m2 = man.add_mod(MyModule(4, 'm2   ', man.port_data, man.port_ctrl))
    m3 = man.add_mod(MyModule(3, 'm3   ', man.port_data, man.port_ctrl))
    m4 = man.add_mod(MyModule(2, 'm4   ', man.port_data, man.port_ctrl))
    
    conn12 = BaseConnectivity(2, 4, 1, m1.id, m2.id)
    conn12[m1.id, :, m2.id, :] = np.ones((2, 4))
    conn12[m2.id, :, m1.id, :] = np.ones((4, 2))
    man.connect(m1, m2, conn12)

    conn23 = BaseConnectivity(4, 3, 1, m2.id, m3.id)
    conn23[m2.id, :, m3.id, :] = np.ones((4, 3))
    conn23[m3.id, :, m2.id, :] = np.ones((3, 4))
    man.connect(m2, m3, conn23)

    conn34 = BaseConnectivity(3, 2, 1, m3.id, m4.id)
    conn34[m3.id, :, m4.id, :] = np.ones((3, 2))
    conn34[m4.id, :, m3.id, :] = np.ones((2, 3))
    man.connect(m3, m4, conn34)

    conn41 = BaseConnectivity(2, 2, 1, m4.id, m1.id)
    conn41[m4.id, :, m1.id, :] = np.ones((2, 2))
    conn41[m1.id, :, m4.id, :] = np.ones((2, 2))
    man.connect(m4, m1, conn41)

    # Start emulation and allow it to run for a little while before shutting
    # down.  To set the emulation to exit after executing a fixed number of
    # steps, start it as follows and remove the sleep statement:
    # man.start(steps=500)
    man.start()
    time.sleep(3)
    man.stop()
    logger.info('all done')
