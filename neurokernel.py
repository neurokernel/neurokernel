#!/usr/bin/env python

"""
Classes for broker-based network of modules controlled by a manager.

Notes
-----
All major object instances are assigned UIDs using
Python's builtin id() function. Since these object instance IDs are
only unique for instantiated objects, generated unique identifiers
should eventually used if the dynamic creation and destruction of
modules must eventually be supported.

"""

import copy, os, signal, sys, threading, time
import multiprocessing as mp
import cPickle as pickle
from contextlib import contextmanager

import twiggy

import numpy as np
import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
import bidict

from nk_uid import uid
from routing_table import RoutingTable
from ctx_managers import IgnoreKeyboardInterrupt, OnKeyboardInterrupt, \
     ExceptionOnSignal, TryExceptionOnSignal
from ctrl_proc import ControlledProcess, LINGER_TIME

PORT_DATA = 5000
PORT_CTRL = 5001

def is_poll_in(sock, poller, timeout=100):
    """
    Check whether a poller detects incoming data on a specified
    socket.
    """
    socks = dict(poller.poll(timeout))
    if sock in socks and socks[sock] == zmq.POLLIN:
        return True
    else:
        return False

class Module(ControlledProcess):
    """
    Processing module.

    This class repeatedly executes a work method until it receives a
    termination signal.

    Parameters
    ----------
    net: str
        Network connectivity. May be `unconnected` for no connection,
        `ctrl` for incoming control data only,
        `in` for incoming data, `out` for outgoing data, or
        `full` for both incoming and outgoing data.
    port_data : int
        Port to use when communicating with broker.
    port_ctrl : int
        Port used by broker to control module.

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

    """

    # Define properties to perform validation when connectivity status
    # is set:
    _net = 'unconnected'
    @property
    def net(self):
        """
        Network connectivity.
        """
        return self._net
    @net.setter
    def net(self, value):
        if value not in ['unconnected', 'ctrl', 'in', 'out', 'full']:
            raise ValueError('invalid network connectivity value')
        self.logger.info('net status changed: %s -> %s' % (self._net, value))
        self._net = value

    def __init__(self, net='unconnected',
                 port_data=PORT_DATA, port_ctrl=PORT_CTRL):
        super(Module, self).__init__(port_ctrl,
                                     signal.SIGUSR1)

        # Logging:
        self.logger = twiggy.log.name('module %s' % self.id)

        # Network connection type:
        self.net = net

        # Data port:
        if port_data == port_ctrl:
            raise ValueError('data and control ports must differ')
        self.port_data = port_data

        # Flag indicating when the module instance is running:
        self.running = False

        # Lists used for storing incoming and outgoing data; each
        # entry is a tuple whose first entry is the source or destination
        # module ID and whose second entry is the data:
        self.in_data = []
        self.out_data = []

        # Lists of incoming and outgoing module IDs:
        self.in_ids = []
        self.out_ids = []

    def _ctrl_handler(self, msg):
        """
        Control port handler.
        """

        self.logger.info('recv: %s' % str(msg))
        if msg[0] == 'quit':
            try:
                self.stream_ctrl.flush()
                self.stream_ctrl.stop_on_recv()
                self.ioloop_ctrl.stop()
            except IOError:
                self.logger.info('streams already closed')
            except:
                self.logger.info('other error occurred')
            self.logger.info('issuing signal %s' % self.quit_sig)
            self.sock_ctrl.send('ack')
            os.kill(os.getpid(), self.quit_sig)

    def _init_net(self):
        """
        Initialize network connection.
        """

        if self.net == 'unconnected':
            self.logger.info('not initializing network connection')
        else:

            # Don't allow interrupts to prevent the handler from
            # completely executing each time it is called:
            with IgnoreKeyboardInterrupt():
                self.logger.info('initializing network connection')

                # Initialize control port handler:
                super(Module, self)._init_net()

                # Use a nonblocking port for the data interface; set
                # the linger period to prevent hanging on unsent
                # messages when shutting down:
                self.sock_data = self.ctx.socket(zmq.DEALER)
                self.sock_data.setsockopt(zmq.IDENTITY, self.id)
                self.sock_data.setsockopt(zmq.LINGER, LINGER_TIME)
                self.sock_data.connect("tcp://localhost:%i" % self.port_data)

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

        if self.net in ['unconnected', 'ctrl']:
            self.logger.info('not synchronizing with network')
            if self.net == 'ctrl' and not self.running:
                return
        else:
            self.logger.info('synchronizing with network')

            if self.net in ['out', 'full']:
                ## should check to make sure that out_data contains
                ## entries for all IDs in self.out_ids
                for out_id, data in self.out_data:
                    self.sock_data.send(pickle.dumps((out_id, data)))
                    self.logger.info('sent to   %s: %s' % (out_id, str(data)))
                self.logger.info('sent data to all output IDs')

            if self.net in ['in', 'full']:
                recv_ids = copy.copy(self.in_ids)
                self.in_data = []
                while recv_ids:
                    in_id, data = pickle.loads(self.sock_data.recv())
                    self.logger.info('recv from %s: %s ' % (in_id, str(data)))
                    recv_ids.remove(in_id)
                    self.in_data.append((in_id, data))
                self.logger.info('recv data from all input IDs')

    def run_step(self):
        """
        This method should be implemented to do something interesting.

        Notes
        -----
        Assumes that the attributes used for input and output already
        exist.

        """

        self.logger.info('running execution step')

        # Create some random data:
        if self.net in ['out', 'full']:
            self.out_data = []
            for i in self.out_ids:
                self.out_data.append((i, str(np.random.rand())))

    def run(self):
        """
        Body of process.
        """

        with TryExceptionOnSignal(self.quit_sig, Exception, self.id):

            # Don't allow keyboard interruption of process:
            with IgnoreKeyboardInterrupt():

                self._init_net()
                np.random.seed()
                self.running = True
                while True:

                    # Run the processing step:
                    self.run_step()

                    # Synchronize:
                    self._sync()

            self.logger.info('exiting')

class Broker(ControlledProcess):
    """
    Broker for communicating between modules.

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
        super(Broker, self).__init__(port_ctrl, signal.SIGUSR1)

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
        self.recv_coords_list = routing_table.coords

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
            except:
                self.logger.info('other error occurred')
            self.sock_ctrl.send('ack')
            #self.logger.info('issuing signal %s' % self.quit_sig)
            #os.kill(os.getpid(), self.quit_sig)

    def _data_handler(self, msg):
        """
        Data port handler.
        """

        if len(msg) != 2:
            self.logger.info('skipping malformed message: %s' % str(msg))
        else:

            # The first entry of the message is the originating ID
            # (prepended by zmq); the second is the destination ID:
            in_id = msg[0]
            out_id, data = pickle.loads(msg[1])
            self.logger.info('recv from %s: %s' % (in_id, data))
            self.logger.info('recv coords list len: '+ str(len(self.recv_coords_list)))
            if (in_id, out_id) in self.recv_coords_list:
                self.data_to_route.append((in_id, out_id, data))
                self.recv_coords_list.remove((in_id, out_id))

            # When data with source/destination IDs corresponding to
            # every entry in the routing table has been received,
            # deliver the data:
            if not self.recv_coords_list:
                self.logger.info('recv from all modules')
                for in_id, out_id, data in self.data_to_route:
                    self.logger.info('sent to   %s: %s' % (out_id, data))

                    # Route to the destination ID and send the source ID
                    # along with the data:
                    self.sock_data.send_multipart([out_id,
                                                   pickle.dumps((in_id, data))])

                # Reset the incoming data buffer and list of connection
                # coordinates:
                self.data_to_route = []
                self.recv_coords_list = self.routing_table.coords
                self.logger.info('----------------------')

    def _init_ctrl_handler(self):
        """
        Initialize control port handler.
        """

        # Set the linger period to prevent hanging on unsent messages
        # when shutting down:
        self.logger.info('initializing ctrl handler')
        self.sock_ctrl = self.ctx.socket(zmq.DEALER)
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
        self.sock_data = self.ctx.socket(zmq.ROUTER)
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
        self.ctx = zmq.Context()
        self.ioloop = IOLoop.instance()
        self._init_ctrl_handler()
        self._init_data_handler()
        self.ioloop.start()

    def run(self):
        """
        Body of process.
        """

        with TryExceptionOnSignal(self.quit_sig, Exception, self.id):
            self.recv_coords_list = self.routing_table.coords
            self._init_net()
        self.logger.info('exiting')

class Connectivity(object):
    """
    Intermodule connectivity class.

    """

    def __init__(self):

        # Unique object ID:
        self.id = uid()

class Manager(object):
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
        self.ctx = zmq.Context()
        self.sock_ctrl = self.ctx.socket(zmq.ROUTER)
        self.sock_ctrl.setsockopt(zmq.LINGER, LINGER_TIME)
        self.sock_ctrl.bind("tcp://*:%i" % self.port_ctrl)

        # Data structures for storing broker, module, and connectivity instances:
        self.broks = bidict.bidict()
        self.mods = bidict.bidict()
        self.conns = bidict.bidict()

        # Set up a dynamic table to contain the routing table:
        self.routing_table = RoutingTable()

    def connect(self, m_src, m_dest, conn):
        """
        Connect two module instances with a connectivity object instance.

        Parameters
        ----------
        m_src : Module
           Source module instance.
        m_dest : Module
           Destination module instance.
        conn : Connectivity
           Connectivity object instance.

        """

        if not isinstance(m_src, Module) or not isinstance(m_dest, Module) or \
            not isinstance(conn, Connectivity):
            raise ValueError('invalid types')

        # Add the module and connection instances to the internal
        # dictionaries of instances if they are not already there:
        if m_src not in self.mods:
            self.add_mod(m_src)
        if m_dest not in self.mods:
            self.add_mod(m_dest)
        if conn not in self.conns:
            self.add_conn(conn)

        # Add the connection to the routing table:
        self.routing_table[m_src.id, m_dest.id] = 1

        # Update the network connectivity of the source and
        # destination module instances if necessary:
        if m_src.net == 'unconnected':
            m_src.net = 'out'
        if m_src.net == 'in':
            m_src.net = 'full'
        if m_dest.net == 'unconnected':
            m_dest.net = 'in'
        if m_dest.net == 'out':
            m_dest.net = 'full'

        # Update each module's lists of incoming and outgoing modules:
        m_src.in_ids = self.routing_table.row_ids(m_src.id)
        m_src.out_ids = self.routing_table.col_ids(m_src.id)
        m_dest.in_ids = self.routing_table.row_ids(m_dest.id)
        m_dest.out_ids = self.routing_table.col_ids(m_dest.id)

    @property
    def N_brok(self):
        """
        Number of brokers.
        """
        return len(self.broks)

    @property
    def N_mod(self):
        """
        Number of modules.
        """
        return len(self.mods)

    @property
    def N_conn(self):
        """
        Number of connectivity objects.
        """

        return len(self.conns)

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
        self.broks[b.id] = b
        self.logger.info('added broker %s' % b.id)
        return b

    def add_mod(self, m=None):
        """
        Add or create a module instance to the emulation.
        """

        if not isinstance(m, Module):
            m = Module(port_data=self.port_data, port_ctrl=self.port_ctrl)
        self.mods[m.id] = m
        self.logger.info('added module %s' % m.id)
        return m

    def add_conn(self, c=None):
        """
        Add or create a connectivity instance to the emulation.
        """

        if not isinstance(c, Connectivity):
            c = Connectivity()
        self.conns[c.id] = c
        self.logger.info('added connectivity %s' % c.id)
        return c

    def start(self):
        """
        Start execution of all processes.
        """

        with IgnoreKeyboardInterrupt():
            for b in self.broks.values():
                b.start()
            for m in self.mods.values():
                m.start();

    def stop(self):
        """
        Stop execution of all processes.
        """

        self.logger.info('stopping all processes')
        poller = zmq.Poller()
        poller.register(self.sock_ctrl, zmq.POLLIN)
        recv_ids = self.mods.keys()
        while recv_ids:

            # Send quit messages and wait for acknowledgments:
            i = recv_ids[0]
            self.logger.info('sent to   %s: quit' % i)
            self.sock_ctrl.send_multipart([i, 'quit'])
            if is_poll_in(self.sock_ctrl, poller):
                 j, data = self.sock_ctrl.recv_multipart()
                 self.logger.info('recv fr   %s: ack' % j)
                 if j in recv_ids:
                     recv_ids.remove(j)
                     self.mods[j].join(1)
        self.logger.info('all modules stopped')

        # After all modules have been stopped, shut down the broker:
        for i in self.broks.keys():
            self.logger.info('sent to   %s: quit' % i)
            self.sock_ctrl.send_multipart([i, 'quit'])
            self.broks[i].join(1)
        self.logger.info('all brokers stopped')

if __name__ == '__main__':

    # Set up logging:
    screen_output = twiggy.outputs.StreamOutput(twiggy.formats.line_format,
                                                stream=sys.stdout)
    file_output = twiggy.outputs.FileOutput('exec.log',
                                            twiggy.formats.line_format,
                                            'w')
    twiggy.addEmitters(('screen', twiggy.levels.DEBUG, None, screen_output),
                       ('file', twiggy.levels.DEBUG, None, file_output))
    logger = twiggy.log.name(('{name:%s}' % 12).format(name='main'))

    # Set up and start emulation:
    man = Manager()
    man.add_brok()
    #m1 = man.add_mod(Module(net='ctrl'))
    #m2 = man.add_mod(Module(net='ctrl'))
    #m3 = man.add_mod(Module(net='ctrl'))
    #m4 = man.add_mod(Module(net='ctrl'))
    m_list = [man.add_mod() for i in xrange(3)]
    # m1 = man.add_mod()
    # m2 = man.add_mod()
    # m3 = man.add_mod()
    # m4 = man.add_mod()
    conn = man.add_conn()
    # man.connect(m1, m2, conn)
    # man.connect(m2, m1, conn)
    # man.connect(m2, m3, conn)
    # man.connect(m3, m2, conn)
    # man.connect(m3, m4, conn)
    # man.connect(m4, m3, conn)
    # man.connect(m4, m1, conn)
    # man.connect(m1, m4, conn)
    for m1, m2 in zip(m_list, [m_list[-1]]+m_list[:-1]):
        man.connect(m1, m2, conn)

    man.start()
    time.sleep(5)
    man.stop()
    logger.info('all done')
