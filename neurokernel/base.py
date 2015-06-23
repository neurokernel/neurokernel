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
import time
import collections

import bidict
import numpy as np
import twiggy
import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
import msgpack_numpy as msgpack

from mixins import LoggerMixin
from ctrl_proc import ControlledProcess, LINGER_TIME
from ctx_managers import IgnoreKeyboardInterrupt, OnKeyboardInterrupt, \
     ExceptionOnSignal, TryExceptionOnSignal
from tools.comm import get_random_port, sync_router, sync_dealer
from tools.logging import setup_logger
from routing_table import RoutingTable
from uid import uid
from tools.misc import catch_exception
from pattern import Interface, Pattern
from plsel import SelectorMethods, PortMapper

PORT_DATA = 5000
PORT_CTRL = 5001
PORT_TIME = 5002

POLL_TIMEOUT = 100

class BaseModule(ControlledProcess):
    """
    Processing module.

    This class repeatedly executes a work method until it receives a
    quit message via its control network port. 

    Parameters
    ----------
    sel : str, unicode, or sequence
        Path-like selector describing the module's interface of 
        exposed ports.
    sel_in : str, unicode, or sequence
        Selector describing all input ports in the module's interface.
    sel_out : str, unicode, or sequence
        Selector describing all input ports in the module's interface.
    data : numpy.ndarray
        Data array to associate with ports. Array length must equal the number
        of ports in a module's interface.    
    columns : list of str
        Interface port attributes.
    port_data : int
        Network port for transmitting data.
    port_ctrl : int
        Network port for controlling the module instance.
    id : str
        Module identifier. If no identifier is specified, a unique 
        identifier is automatically generated.
    debug : bool
        Debug flag. When True, exceptions raised during the work method
        are not be suppressed.
    time_sync : bool
        Time synchronization flag. When True, debug messages are not emitted during
        module synchronization and the time taken to receive all incoming data is 
        computed.

    Attributes
    ----------
    interface : Interface
        Object containing information about a module's ports.    
    patterns : dict of Pattern
        Pattern objects connecting the module instance with 
        other module instances. Keyed on the ID of the other module 
        instances.
    pat_ints : dict of tuple of int
        Interface of each pattern that is connected to the module instance.
        Keyed on the ID of the other module instances.
    pm : plsel.PortMapper
        Map between a module's ports and the contents of the `data` attribute.
    data : numpy.ndarray
        Array of data associated with a module's ports.

    Notes
    -----
    If the network ports specified upon instantiation are None, the module
    instance ignores the network entirely.
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
        self.log_info('net status changed: %s -> %s' % (self._net, value))
        self._net = value

    # Define properties to perform validation when the maximum number of
    # execution steps set:
    _max_steps = float('inf')
    @property
    def max_steps(self):
        """
        Maximum number of steps to execute.
        """
        return self._max_steps
    @max_steps.setter
    def max_steps(self, value):
        if value <= 0:
            raise ValueError('invalid maximum number of steps')
        self.log_info('maximum number of steps changed: %s -> %s' % \
                        (self._max_steps, value))
        self._max_steps = value

    # Define properties to perform validation when the synchronization 
    # period is set:
    _sync_period = 0
    @property
    def sync_period(self):
        """
        How many steps between synchronization.
        """
        return self._sync_period
    @sync_period.setter
    def sync_period(self, value):
        if value < 0:
            raise ValueError('invalid synchronization period')
        self.log_info('synchronization period changed: %s -> %s' % \
                      (self._sync_period, value))
        self._sync_period = value

    def __init__(self, sel, sel_in, sel_out,
                 data, columns=['interface', 'io', 'type'],
                 port_data=PORT_DATA, port_ctrl=PORT_CTRL, port_time=PORT_TIME,
                 id=None, debug=False, time_sync=False):
        self.debug = debug
        self.time_sync = time_sync

        # Require several necessary attribute columns:
        assert 'interface' in columns
        assert 'io' in columns
        assert 'type' in columns

        # Generate a unique ID if none is specified:
        if id is None:
            id = uid()

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

        # Set the interface ID to 0; we assume that a module only has one
        # interface:
        self.interface[sel, 'interface'] = 0

        # Set port I/O status:
        assert SelectorMethods.is_in(sel_in, sel)
        assert SelectorMethods.is_in(sel_out, sel)
        assert SelectorMethods.are_disjoint(sel_in, sel_out)
        self.interface[sel_in, 'io'] = 'in'
        self.interface[sel_out, 'io'] = 'out'

        # Set up mapper between port identifiers and their associated data:
        assert len(data) == len(self.interface)
        self.data = data
        self.pm = PortMapper(sel, self.data)

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

        # Dictionary containing ports of source modules that
        # send output to this module. Must be initialized immediately before
        # an emulation begins running. Keyed on source module ID:
        self._in_port_dict = {}

        # Dictionary containing ports of destination modules that
        # receive input from this module. Must be initialized immediately before
        # an emulation begins running. Keyed on destination module ID:
        self._out_port_dict = {}

        self._out_ids = []
        self._in_ids = []
        
    @property
    def N_ports(self):
        """
        Number of ports exposed by module's interface.
        """

        return len(self.interface.ports())

    @property
    def all_ids(self):
        """
        IDs of modules to which the current module is connected.
        """

        return self.patterns.keys()

    @property
    def in_ids(self):
        """
        IDs of modules that send data to this module.
        """

        return [m for m in self.patterns.keys() \
                if self.patterns[m].is_connected(self.pat_ints[m][1],
                                                 self.pat_ints[m][0])]

    @property
    def out_ids(self):
        """
        IDs of modules that receive data from this module.
        """

        return [m for m in self.patterns.keys() \
                if self.patterns[m].is_connected(self.pat_ints[m][0],
                                                 self.pat_ints[m][1])]

    def connect(self, m, pat, int_0, int_1, compat_check=True):
        """
        Connect the current module instance to another module with a pattern instance.

        Parameters
        ----------
        m : BaseModule
            Module instance to connect.
        pat : Pattern
            Pattern instance.
        int_0, int_1 : int
            Which of the pattern's interface to connect to the current module
            and the specified module, respectively.
        compat_check : bool        
            Check whether the interfaces of the current and specified modules
            are compatible with the specified pattern. This option is provided
            because compatibility checking can be expensive.
        """

        assert isinstance(m, BaseModule)
        assert isinstance(pat, Pattern)
        assert int_0 in pat.interface_ids and int_1 in pat.interface_ids
        self.log_info('connecting to %s' % m.id)

        # Check compatibility of the interfaces exposed by the modules and the
        # pattern:
        if compat_check:
            self.log_info('checking compatibility of modules {0} and {1} and'
                             ' assigned pattern'.format(self.id, m.id))
            assert self.interface.is_compatible(0, pat.interface, int_0, True)
            assert m.interface.is_compatible(0, pat.interface, int_1, True)

        # Check that no fan-in from different source modules occurs as a result
        # of the new connection by getting the union of all connected input
        # ports for the interfaces of all existing patterns connected to the
        # current module and ensuring that the input ports from the new pattern
        # don't overlap:
        if self.patterns:
            curr_in_ports = reduce(set.union,
                [set(self.patterns[i].connected_ports(self.pat_ints[i][0]).in_ports(tuples=True)) \
                     for i in self.patterns.keys()])
            assert not curr_in_ports.intersection(pat.connected_ports(int_0).in_ports(tuples=True))

        # The pattern instances associated with the current
        # module are keyed on the IDs of the modules to which they connect:
        self.patterns[m.id] = pat
        self.pat_ints[m.id] = (int_0, int_1)

        # Update internal connectivity based upon contents of connectivity
        # object. When this method is invoked, the module's internal
        # connectivity is always upgraded to at least 'ctrl':
        if self.net == 'none':
            self.net = 'ctrl'
        if pat.is_connected(int_0, int_1):
            old_net = self.net
            if self.net == 'ctrl':
                self.net = 'out'
            elif self.net == 'in':
                self.net = 'full'
            self.log_info('net status changed: %s -> %s' % (old_net, self.net))
        if pat.is_connected(int_1, int_0):
            old_net = self.net
            if self.net == 'ctrl':
                self.net = 'in'
            elif self.net == 'out':
                self.net = 'full'
            self.log_info('net status changed: %s -> %s' % (old_net, self.net))

    def _ctrl_stream_shutdown(self):
        """
        Shut down control port handler's stream and ioloop.
        """

        try:
            self.stream_ctrl.flush()
            self.stream_ctrl.stop_on_recv()
            self.ioloop_ctrl.stop()
        except IOError:
            self.log_info('streams already closed')
        except:
            self.log_info('other error occurred')
        else:
            self.log_info('ctrl stream shut down')

    def _ctrl_handler(self, msg):
        """
        Control port handler.
        """

        self.log_info('recv ctrl message: %s' % str(msg))
        if msg[0] == 'quit':
            self._ctrl_stream_shutdown()

            # Force the module's main loop to exit:
            self.running = False
            ack = 'shutdown'

        # One can define additional messages to be recognized by the control
        # handler:        
        # elif msg[0] == 'conn':
        #     self.log_info('conn payload: '+str(msgpack.unpackb(msg[1])))
        #     ack = 'ack'
        else:
            ack = 'ack'

        self.sock_ctrl.send(ack)
        self.log_info('sent to manager: %s' % ack)

    def _init_net(self):
        """
        Initialize network connection.
        """

        # Initialize control port handler:
        self.log_info('initializing ctrl network connection')
        super(BaseModule, self)._init_net()

        # Initialize data port handler:
        if self.net == 'none':
            self.log_info('not initializing data network connection')
        else:

            # Don't allow interrupts to prevent the handler from
            # completely executing each time it is called:
            with IgnoreKeyboardInterrupt():
                self.log_info('initializing data network connection')

                # Use a nonblocking port for the data interface; set
                # the linger period to prevent hanging on unsent
                # messages when shutting down:
                self.sock_data = self.zmq_ctx.socket(zmq.DEALER)
                self.sock_data.setsockopt(zmq.IDENTITY, self.id)
                self.sock_data.setsockopt(zmq.LINGER, LINGER_TIME)
                self.sock_data.connect("tcp://localhost:%i" % self.port_data)
                self.log_info('data network connection initialized')

                # Initialize timing port:
                self.log_info('initializing time port')
                self.sock_time = self.zmq_ctx.socket(zmq.DEALER)
                self.sock_time.setsockopt(zmq.IDENTITY, self.id)
                self.sock_data.setsockopt(zmq.LINGER, LINGER_TIME)
                self.sock_time.connect("tcp://localhost:%i" % self.port_time)
                sync_dealer(self.sock_time, self.id)
                self.log_info('time port initialized')

    def _get_in_data(self):
        """
        Get input data from incoming transmission buffer.

        Populate the data array associated with a module's ports using input
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
                    self.pm.set_by_inds(self._in_port_dict_ids[in_id],
                                        self._in_data[in_id].popleft())
                except:
                    self.log_info('no input data from [%s] retrieved' % in_id)
                else:
                    self.log_info('input data from [%s] retrieved' % in_id)

    def _put_out_data(self):
        """
        Put output data in outgoing transmission buffer.

        Stage data from the data array associated with a module's ports for
        output to other modules.
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
                try:
                    data = self.pm.get_by_inds(self._out_port_dict_ids[out_id])
                    self._out_data.append((out_id, data))
                except:
                    self.log_info('no output data to [%s] sent' % out_id)
                else:
                    self.log_info('output data to [%s] sent' % out_id)

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
            self.log_info('not synchronizing with network')
        else:
            self.log_info('synchronizing with network')

            # Send outbound data:
            start = time.time()
            self._put_out_data()
            if self.net in ['out', 'full']:

                # Send all data in outbound buffer:
                send_ids = [out_id for out_id in self._out_ids]
                for out_id, data in self._out_data:
                    self.sock_data.send(msgpack.packb((out_id, data)))
                    send_ids.remove(out_id)
                    if not self.time_sync:
                        self.log_info('sent to   %s: %s' % (out_id, str(data)))

                # Send data tuples containing None to those modules for which no
                # actual data was generated to satisfy the barrier condition:
                for out_id in send_ids:
                    self.sock_data.send(msgpack.packb((out_id, None)))
                    if not self.time_sync:
                        self.log_info('sent to   %s: %s' % (out_id, None))

                # All output IDs should be sent data by this point:
                if not self.time_sync:
                    self.log_info('sent data to all output IDs')

            # Receive inbound data:
            if self.net in ['in', 'full']:

                # Wait until inbound data is received from all source modules:
                recv_ids = set(self._in_ids)
                nbytes = 0
                while recv_ids:

                    # Poll to avoid blocking:
                    if self.sock_data.poll(POLL_TIMEOUT):
                        data_packed = self.sock_data.recv()
                        in_id, data = msgpack.unpackb(data_packed)
                        if not self.time_sync:
                            self.log_info('recv from %s: %s' % (in_id, str(data)))

                        # Ignore incoming data containing None:
                        if data is not None:
                            self._in_data[in_id].append(data)

                            # Record number of bytes of transmitted serialized data:
                            nbytes += len(data_packed)

			# Remove source module ID from set of IDs from which to
                        # expect data:
                        recv_ids.discard(in_id)

                    # Stop the synchronization if a quit message has been received:
                    if not self.running:
                        if not self.time_sync:
                            self.log_info('run loop stopped - stopping sync')
                        break

                if not self.time_sync:
                    self.log_info('recv data from all input IDs')
            self._get_in_data()

            # Transmit synchronization time:
            stop = time.time()
            if self.time_sync:
                self.log_info('sent timing data to master')
                self.sock_time.send(msgpack.packb((self.id, self.steps, 'sync',
                                                   (start, stop, nbytes))))

    def pre_run(self, *args, **kwargs):
        """
        Code to run before main module run loop.

        Code in this method will be executed after a module's process has been
        launched and all connectivity objects made available, but before the
        main run loop begins.
        """

        self.log_info('performing pre-emulation operations')

    def post_run(self, *args, **kwargs):
        """
        Code to run after main module run loop.

        Code in this method will be executed after a module's main loop has
        terminated.
        """

        self.log_info('performing post-emulation operations')

    def run_step(self):
        """
        Module work method.

        This method should be implemented to do something interesting with new 
        input port data in the module's `pm` attribute and update the attribute's
        output port data if necessary. It should not interact with any other 
        class attributes.
        """

        self.log_info('running execution step')

    def post_run_step(self):
        """
        Code to run after each execution step.

        This method can be implemented to do something immediately after each
        invocation of `self.run_step()`, e.g., save generated data to a file, etc.
        """

        pass

    def _init_port_dicts(self):
        """
        Initial dictionaries of source/destination ports in current module.
        """

        # Extract identifiers of source ports in the current module's interface
        # for all modules receiving output from the current module:
        self._out_port_dict = {}
        self._out_port_dict_ids = {}
        self._out_ids = self.out_ids
        for out_id in self._out_ids:
            self.log_info('extracting output ports for %s' % out_id)

            # Get interfaces of pattern connecting the current module to
            # destination module `out_id`; `from_int` is connected to the
            # current module, `to_int` is connected to the other module:
            from_int, to_int = self.pat_ints[out_id]

            # Get ports in interface (`from_int`) connected to the current
            # module that are connected to the other module via the pattern:
            self._out_port_dict[out_id] = \
                self.patterns[out_id].src_idx(from_int, to_int)
            self._out_port_dict_ids[out_id] = \
                self.pm.ports_to_inds(self._out_port_dict[out_id])

        # Extract identifiers of destination ports in the current module's
        # interface for all modules sending input to the current module:
        self._in_port_dict = {}
        self._in_port_dict_ids = {}
        self._in_ids = self.in_ids
        for in_id in self._in_ids:
            self.log_info('extracting input ports for %s' % in_id)

            # Get interfaces of pattern connecting the current module to
            # source module `out_id`; `to_int` is connected to the current
            # module, `from_int` is connected to the other module:
            to_int, from_int = self.pat_ints[in_id]

            # Get ports in interface (`to_int`) connected to the current
            # module that are connected to the other module via the pattern:
            self._in_port_dict[in_id] = \
                self.patterns[in_id].dest_idx(from_int, to_int)
            self._in_port_dict_ids[in_id] = \
                self.pm.ports_to_inds(self._in_port_dict[in_id])

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
            # queue buffering the received data:
            self._in_data = {k: collections.deque() for k in self.in_ids}

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

                # Exit run loop when a quit message has been received:
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
    routing_table : routing_table.RoutingTable
        Directed graph of network connections between modules comprised by an
        emulation.
    """

    def __init__(self, port_data=PORT_DATA, port_ctrl=PORT_CTRL,
                 routing_table=None):
        super(Broker, self).__init__(port_ctrl, uid())

        # Reformat logger name:
        LoggerMixin.__init__(self, 'brk %s' % self.id)

        # Data port:
        if port_data == port_ctrl:
            raise ValueError('data and control ports must differ')
        self.port_data = port_data

        # Routing table:
        self.routing_table = routing_table

        # Buffers used to accumulate data to route:
        self._data_to_route = []

    def _ctrl_handler(self, msg):
        """
        Control port handler.
        """

        self.log_info('recv: '+str(msg))
        if msg[0] == 'quit':
            try:
                self.stream_ctrl.flush()
                self.stream_data.flush()
                self.stream_ctrl.stop_on_recv()
                self.stream_data.stop_on_recv()
                self.ioloop.stop()
            except IOError:
                self.log_info('streams already closed')
            except Exception as e:
                self.log_info('other error occurred: '+e.message)
            self.sock_ctrl.send('ack')
            self.log_info('sent to  broker: ack')

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
            self.log_info('skipping malformed message: %s' % str(msg))
        else:

            # Queue arriving messages:
            in_id = msg[0]
            out_id, data = msgpack.unpackb(msg[1])
            self.log_info('recv from %s: %s' % (in_id, data))
            self._recv_queues[(in_id, out_id)].appendleft(data)

            # When data with source/destination IDs corresponding to
            # every entry in the routing table has been received up to
            # the current time step (i.e., all queues for each 
            # source/destination pair contain something), deliver the data:
            if all(self._recv_queues.values()):
                self.log_info('recv from all modules')
                for t in self._recv_queues:
                    in_id, out_id = t
                    data = self._recv_queues[t].pop()
                    self.log_info('sent to   %s: %s' % (out_id, data))

                    # Route to the destination ID and send the source ID
                    # along with the data:
                    self.sock_data.send_multipart([out_id,
                                                   msgpack.packb((in_id, data))])

                self.log_info('----------------------')

    def _init_ctrl_handler(self):
        """
        Initialize control port handler.
        """

        # Set the linger period to prevent hanging on unsent messages
        # when shutting down:
        self.log_info('initializing ctrl handler')
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
        self.log_info('initializing data handler')
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
        self.log_info('starting')
        with IgnoreKeyboardInterrupt():
            self._recv_queues = \
                {c:collections.deque() for c in \
                 self.routing_table.connections}
            self._init_net()
        self.log_info('exiting')

class TimeListener(ControlledProcess):
    """
    Class for collecting/processing emitted module timing data.

    Parameters
    ----------
    port_ctrl : int
        Network port for controlling the module instance.
    port_time : int
        Network port for receiving transmitted timing data.
    ids : set of str
        Set of module IDs from which to receive timing data.

    Notes
    -----
    The IDs of all modules from which timing data will be collected
    must be stored in the class instance's `ids` attribute before
    the process is started.

    This class should only be instantiated by the Manager class.
    """

    def __init__(self, port_ctrl, port_time, ids=set()):
        super(TimeListener, self).__init__(port_ctrl, uid())

        # Reformat logger name:
        LoggerMixin.__init__(self, 'lis %s' % self.id)

        # Time port:
        if port_time == port_ctrl:
            raise ValueError('time and control ports must differ')
        self.port_time = port_time

        # IDs of modules from which to collect timing data:
        assert isinstance(ids, set)
        self.ids = ids

        # Queue for returning timing results to parent process:
        self.queue = mp.Queue()

    def add(self, id):
        """
        Add a module ID from which to collect timing data.
        """

        self.ids.add(id)

    def run(self):
        self._init_net()
        sock_time = self.zmq_ctx.socket(zmq.ROUTER)
        sock_time.bind('tcp://*:%s' % self.port_time)
        sync_router(sock_time, self.ids)
        self.log_info('time port initialized')
        self.running = True
        counter = 0
        total_sync_time = 0.0
        total_sync_nbytes = 0.0
        received_data = {}
        self.start_time = 0.0
        self.stop_time = 0.0
        self.average_throughput = 0.0
        self.average_step_sync_time = 0.0
        while True:
            if sock_time.poll(10):

                # Receive timing data:
                id, data = sock_time.recv_multipart()
                id, steps, time_type, data = msgpack.unpackb(data)
                self.log_info('time data: %s, %s, %s, %s' % (id, steps,
                                                             time_type, str(data)))

                # The time_type may be 'start' (emulation run loop start time), 
                # 'stop' (emulation loop stop time), or 'sync' (emulation sync
                # time data):
                if time_type == 'start':
                    self.start_time = data
                elif time_type == 'stop':
                    self.stop_time = data
                elif time_type == 'sync':
                    start, stop, nbytes = data

                    # Collect timing data for each execution step:
                    if steps not in received_data:
                        received_data[steps] = {}                    
                    received_data[steps][id] = (start, stop, nbytes)

                    # After adding the latest timing data for a specific step, check
                    # whether data from all modules has arrived for that step:
                    if set(received_data[steps].keys()) == self.ids:

                        # The duration an execution is assumed to be the longest of
                        # the received intervals:
                        step_sync_time = max([(d[1]-d[0]) for d in received_data[steps].values()])

                        # Obtain the total number of bytes received by all of the
                        # modules during the execution step:
                        step_nbytes = sum([d[2] for d in received_data[steps].values()])

                        total_sync_time += step_sync_time
                        total_sync_nbytes += step_nbytes

                        self.average_throughput = (self.average_throughput*counter+\
                                                   step_nbytes/step_sync_time)/(counter+1)
                        self.average_step_sync_time = (self.average_step_sync_time*counter+\
                                                       step_sync_time)/(counter+1)

                        # Clear the data for the processed execution step so that
                        # that the received_data dict doesn't consume unnecessary memory:
                        del received_data[steps]

                        counter += 1
                    
            if not self.running:
                self.log_info('stopping run loop')
                break
        self.log_info('done')

        if total_sync_time > 0.0:
            self.total_throughput = total_sync_nbytes/total_sync_time
        else:
            self.total_throughput = 0.0
        self.log_info('avg step sync time/avg per-step throughput' \
                      '/total transm throughput/run loop duration:' \
                      '%s, %s, %s, %s' % \
                      (self.average_step_sync_time, self.average_throughput, 
                       self.total_throughput, self.stop_time-self.start_time))
        self.queue.put((self.average_step_sync_time, self.average_throughput,
                        self.total_throughput, self.stop_time-self.start_time))

    def get_throughput(self):
        """
        Retrieve average step sync time, average per-step throughput, total
        transmission throughput, and run loop duration.
        """

        return self.queue.get()

class BaseManager(LoggerMixin):
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
    port_time : int
        Port used to obtain timing information from modules.

    Attributes
    ----------
    brokers : dict
        Communication brokers. Keyed by broker object ID.
    modules : dict
        Module instances. Keyed by module object ID.
    routing_table : routing_table.RoutingTable
        Table of data transmission connections between modules.
    time_listener : TimeListener
        Process for collecting timing data from all modules.

    Notes
    -----
    The message flow between objects in a Neurokernel emulation is as follows:
    
    Manager -[ctrl]-> BaseModule, Broker, TimeListener
    BaseModule -[time]-> TimeListener
    BaseModule -[data]-> Broker -[data]-> BaseModule
    """ 

    def __init__(self, port_data=PORT_DATA, port_ctrl=PORT_CTRL,
                 port_time=PORT_TIME):

        # Unique object ID:
        self.id = uid()

        # Set logger name:
        LoggerMixin.__init__(self, 'man %s' % self.id)

        self.port_data = port_data
        self.port_ctrl = port_ctrl
        self.port_time = port_time

        # Set up a router socket to communicate with other topology
        # components; linger period is set to 0 to prevent hanging on
        # unsent messages when shutting down:
        self.zmq_ctx = zmq.Context()
        self.sock_ctrl = self.zmq_ctx.socket(zmq.ROUTER)
        self.sock_ctrl.setsockopt(zmq.LINGER, LINGER_TIME)
        self.sock_ctrl.bind("tcp://*:%i" % self.port_ctrl)
        
        # Data structures for instances of objects that correspond to processes
        # keyed on object IDs (bidicts are used to enable retrieval of
        # broker/module IDs from object instances):
        self.brokers = bidict.bidict()
        self.modules = bidict.bidict()

        # Set up a dynamic table to contain the routing table:
        self.routing_table = RoutingTable()

        # Number of emulation steps to run:
        self.max_steps = float('inf')

        # Set up process to handle time data:
        self.time_listener = TimeListener(self.port_ctrl, self.port_time)

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
        self.log_info('passing connection pattern to modules {0} and {1}'
            .format(m_0.id, m_1.id))
        m_0.connect(m_1, pat, int_0, int_1)
        m_1.connect(m_0, pat, int_1, int_0)

        # Update the routing table:
        self.log_info('updating routing table')
        if pat.is_connected(0, 1):
            self.routing_table[m_0.id, m_1.id] = 1
        if pat.is_connected(1, 0):
            self.routing_table[m_1.id, m_0.id] = 1

        self.log_info('connected modules {0} and {1}'.format(m_0.id, m_1.id))

    @property
    def N_brok(self):
        """
        Number of brokers.
        """
        return len(self.brokers)

    @property
    def N_mod(self):
        """
        Number of modules.
        """
        return len(self.modules)


    def add_brok(self, b=None):
        """
        Add or create a broker instance to the emulation.
        """

        # TEMPORARY: only allow one broker:
        if self.N_brok == 1:
            raise RuntimeError('only one broker allowed')

        if not isinstance(b, Broker):
            b = Broker(port_data=self.port_data,
                       port_ctrl=self.port_ctrl,
                       routing_table=self.routing_table)
        self.brokers[b.id] = b
        self.log_info('added broker %s' % b.id)
        return b

    def add_mod(self, m=None):
        """
        Add or create a module instance to the emulation.

        Parameters
        ----------
        m : str
            ID of module to add.
        """

        if not isinstance(m, BaseModule):
            m = BaseModule(port_data=self.port_data,
                           port_ctrl=self.port_ctrl, port_time=self.port_time)
        self.modules[m.id] = m
        self.log_info('added module %s' % m.id)
        return m

    def start(self, steps=np.inf, sync_period=0):
        """
        Start execution of all processes.

        Parameters
        ----------
        steps : number
            Maximum number of steps to execute.
        sync_period : number
            Transmit data between modules every `sync_period` steps, i.e., 
            if 0, transmit data at every step, if 1, transmit data at every
            other step, etc.
        """

        self.max_steps = steps
        self.sync_period = sync_period
        with IgnoreKeyboardInterrupt():
            self.log_info('time listener about to start')
            self.time_listener.start()
            self.log_info('time listener started')
            bi = 1
            mi = 1
            for b in self.brokers.values():
                self.log_info('broker ' + str(bi) + ' about to start')
                b.start()
                self.log_info('broker ' + str(bi) + ' started')
                bi+=1
            for m in self.modules.values():
                m.max_steps = steps
                m.sync_period = sync_period
                self.log_info('module ' + str(mi) + ' about to start')
                m.start()
                self.log_info('module ' + str(mi) + ' started')
                mi+=1

    def send_ctrl_msg(self, i, *msg):
        """
        Send control message(s) to a module.
        """

        self.sock_ctrl.send_multipart([i]+msg)
        self.log_info('sent to   %s: %s' % (i, msg))
        while True:
            if self.sock_ctrl.poll(POLL_TIMEOUT):
                j, data = self.sock_ctrl.recv_multipart()
                self.log_info('recv from %s: ack' % j)
                break

    def stop_brokers(self):
        """
        Stop execution of all brokers.
        """

        self.log_info('stopping all brokers')
        for i in self.brokers.keys():
            self.log_info('sent to   %s: quit' % i)
            self.sock_ctrl.send_multipart([i, 'quit'])
            self.brokers[i].join(1)
        self.log_info('all brokers stopped')
        
    def stop_listener(self):
        """
        Stop execution of the time listener.
        """

        self.log_info('stopping time listener')
        self.log_info('sent to   %s: quit' % self.time_listener.id)
        self.sock_ctrl.send_multipart([self.time_listener.id, 'quit'])
        self.time_listener.join(1)
        self.log_info('time listener stopped')

    def join_modules(self, send_quit=False):
        """
        Wait until all modules have stopped.

        Parameters
        ----------
        send_quit : bool
            If True, send quit messages to all modules.            
        """

        self.log_info('waiting for modules to shut down')
        recv_ids = self.modules.keys()
        while recv_ids:
            i = recv_ids[0]
            
            # Send quit messages to all live modules:
            if send_quit:                
                self.log_info('live modules: '+str(recv_ids))
                self.log_info('sent to   %s: quit' % i)
                self.sock_ctrl.send_multipart([i, 'quit'])

            # If a module acknowledges receiving a quit message,
            # wait for it to shutdown:
            if self.sock_ctrl.poll(POLL_TIMEOUT):
                 j, data = self.sock_ctrl.recv_multipart()
                 self.log_info('recv from %s: %s' % (j, data))                 
                 if j in recv_ids and data == 'shutdown':
                     self.log_info('waiting for module %s to shut down' % j)
                     recv_ids.remove(j)
                     self.modules[j].join(1)
                     self.log_info('module %s shut down' % j)
                     
            # Sometimes quit messages are received but the acknowledgements are
            # lost; if so, the module will eventually shutdown:
            # XXX this shouldn't be necessary XXX
            if not self.modules[i].is_alive() and i in recv_ids:
                self.log_info('%s shutdown without ack' % i)
                recv_ids.remove(i)                
        self.log_info('all modules stopped')

    def stop(self):
        """
        Stop execution of an emulation.
        """

        if np.isinf(self.max_steps):
            self.log_info('stopping all modules')
            send_quit = True
        else:
            send_quit = False
        self.join_modules(send_quit)
        self.stop_brokers()
        self.stop_listener()

    def get_throughput(self):
        """
        Retrieve average received data throughput.
        """

        return self.time_listener.get_throughput()

if __name__ == '__main__':
    from neurokernel.tools.misc import rand_bin_matrix

    class MyModule(BaseModule):
        """
        Example of derived module class.
        """

        def run_step(self):
            super(MyModule, self).run_step()

            # Do something with input data:
            in_ports = self.interface.in_ports().to_tuples()
            self.log_info('input port data: '+str(self.pm[in_ports]))

            # Output random data:
            out_ports = self.interface.out_ports().to_tuples()
            self.pm[out_ports] = np.random.rand(len(out_ports))
            self.log_info('output port data: '+str(self.pm[out_ports]))

    # Set up logging:
    logger = setup_logger(screen=True)

    # Set up emulation:
    man = BaseManager(get_random_port(), get_random_port(), get_random_port())
    man.add_brok()

    m1_int_sel = '/a[0:5]'; m1_int_sel_in = '/a[0:2]'; m1_int_sel_out = '/a[2:5]'
    m2_int_sel = '/b[0:5]'; m2_int_sel_in = '/b[0:3]'; m2_int_sel_out = '/b[3:5]'
    m3_int_sel = '/c[0:4]'; m3_int_sel_in = '/c[0:2]'; m3_int_sel_out = '/c[2:4]'

    m1 = MyModule(m1_int_sel, m1_int_sel_in, m1_int_sel_out,
                  np.zeros(5, dtype=np.float),
                  ['interface', 'io', 'type'],
                  man.port_data, man.port_ctrl, man.port_time, 'm1   ', False,
                  True)
    man.add_mod(m1)
    m2 = MyModule(m2_int_sel, m2_int_sel_in, m2_int_sel_out,
                  np.zeros(5, dtype=np.float),
                  ['interface', 'io', 'type'],
                  man.port_data, man.port_ctrl, man.port_time, 'm2   ', False,
                  True)
    man.add_mod(m2)
    m3 = MyModule(m3_int_sel, m3_int_sel_in, m3_int_sel_out,
                  np.zeros(4, dtype=np.float),
                  ['interface', 'io', 'type'], 
                  man.port_data, man.port_ctrl, man.port_time, 'm3   ', False,
                  True)
    man.add_mod(m3)

    # Make sure that all ports in the patterns' interfaces are set so 
    # that they match those of the modules:
    pat12 = Pattern(m1_int_sel, m2_int_sel)
    pat12.interface[m1_int_sel_out] = [0, 'in']
    pat12.interface[m1_int_sel_in] = [0, 'out']
    pat12.interface[m2_int_sel_in] = [1, 'out']
    pat12.interface[m2_int_sel_out] = [1, 'in']
    pat12['/a[2]', '/b[0]'] = 1
    pat12['/a[3]', '/b[1]'] = 1
    pat12['/b[3]', '/a[0]'] = 1
    man.connect(m1, m2, pat12, 0, 1)

    pat23 = Pattern(m2_int_sel, m3_int_sel)
    pat23.interface[m2_int_sel_out] = [0, 'in']
    pat23.interface[m2_int_sel_in] = [0, 'out']
    pat23.interface[m3_int_sel_in] = [1, 'out']
    pat23.interface[m3_int_sel_out] = [1, 'in']
    pat23['/b[4]', '/c[0]'] = 1
    pat23['/c[2]', '/b[2]'] = 1
    man.connect(m2, m3, pat23, 0, 1)

    pat31 = Pattern(m3_int_sel, m1_int_sel)
    pat31.interface[m3_int_sel_out] = [0, 'in']
    pat31.interface[m1_int_sel_in] = [1, 'out']
    pat31.interface[m3_int_sel_in] = [0, 'out']
    pat31.interface[m1_int_sel_out] = [1, 'in']
    pat31['/c[3]', '/a[1]'] = 1
    pat31['/a[4]', '/c[1]'] = 1
    man.connect(m3, m1, pat31, 0, 1)

    # Start emulation and allow it to run for a little while before shutting
    # down.  To set the emulation to exit after executing a fixed number of
    # steps, start it as follows and remove the sleep statement:
    # man.start(steps=500)

    man.start()
    time.sleep(2)
    man.stop()
    logger.info('all done')
