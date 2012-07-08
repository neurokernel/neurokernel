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

import copy, logging, threading, time
import multiprocessing as mp
import cPickle as pickle
import numpy as np
import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
import bidict

from routing_table import RoutingTable
from int_ctx import NoKeyboardInterrupt, OnKeyboardInterrupt

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

class Module(mp.Process):
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
    ctrl_recv_handler(msg)
        Handle messages received on control interface.
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

        # Use the object instance's Python ID (i.e., memory address)
        # as its UID:
        self.id = str(id(self))

        self.logger = logging.getLogger('module %s' % self.id)

        self.net = net
        self.port_data = port_data
        self.port_ctrl = port_ctrl
        if self.port_data == self.port_ctrl:
            raise ValueError('data and control ports must differ')

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

        super(Module, self).__init__()

    def _ctrl_handler(self, msg):
        """
        Control interface handler for received data.

        Notes
        -----
        Assumes that self.ioloop and self.stream exist.

        """

        data = msg[0]
        if data == 'quit':
            self.logger.info('recv: quit')
            self.stream.flush()
            self.ioloop.stop()
            self.running = False

    def init_net(self):
        """
        Initialize the network connection.
        """

        if self.net == 'unconnected':
            self.logger.info('not initializing network connection')
        else:

            # Don't allow interrupts to prevent the handler from
            # completely executing each time it is called:
            with NoKeyboardInterrupt():
                self.logger.info('initializing network connection')
                self.ctx = zmq.Context()

                # Use a nonblocking port for the control interface:
                self.sock_ctrl = self.ctx.socket(zmq.DEALER)
                self.sock_ctrl.setsockopt(zmq.IDENTITY, self.id)
                self.sock_ctrl.connect("tcp://localhost:%i" % self.port_ctrl)

                # Use a nonblocking port for the data interface:
                self.sock_data = self.ctx.socket(zmq.DEALER)
                self.sock_data.setsockopt(zmq.IDENTITY, self.id)
                self.sock_data.connect("tcp://localhost:%i" % self.port_data)

                # Set up control stream event handler:
                self.ioloop = IOLoop.instance()
                self.stream = ZMQStream(self.sock_ctrl, self.ioloop)
                self.stream.on_recv(self._ctrl_handler)

                # Start the event loop in a separate thread so as to
                # not block execution:
                threading.Thread(target=self.ioloop.start).start()

    def _sync(self):
        """
        Send output data and receive input data.

        Notes
        -----
        Assumes that the attributes used for input and output already
        exist.

        Data is serialized before being sent and unserialized when
        received.

        Returns
        -------
        done : bool
            Set to True when it is time to quit.
            
        """

        done = False
        if self.net == 'unconnected':
            self.logger.info('not synchronizing with network')
        else:
            self.logger.info('synchronizing with network')

            if self.net in ['out', 'full']:

                ## should check to make sure that out_data contains
                ## entries for all IDs in self.out_ids
                for out_id, data in self.out_data:
                    self.sock_data.send_multipart([out_id, pickle.dumps(data)])
                    self.logger.info('sent to   %s: %s' % (out_id, str(data)))
                self.logger.info('sent data to all output IDs')
                if self.net == 'out' and not self.running:
                    for i in xrange(50):
                        self.send_null()
                    return True
            if self.net in ['in', 'full']:
                recv_ids = copy.copy(self.in_ids)
                self.in_data = []
                while recv_ids:
                    if not self.running:
                        self.logger.info('setting timeout')
                        self.sock_data.setsockopt(zmq.RCVTIMEO, 0)
                    try:
                        in_id, data = self.sock_data.recv_multipart()
                    except zmq.ZMQError:
                        self.logger.info('timeout forced')
                        done = True
                        return done
                    data = pickle.loads(data)
                    self.logger.info('recv from %s: %s ' % (in_id, str(data)))
                    recv_ids.remove(in_id)
                    self.in_data.append((in_id, data))
                    self.logger.info('recv data from all input IDs')
        return done                 

    def send_null(self):
        if self.net in ['out', 'full']:
            self.logger.info('sending null data')
            for out_id in self.out_ids:
                self.sock_data.send_multipart([out_id, pickle.dumps('')])
                self.logger.info('sent to   %s: NULL' % out_id)
                self.logger.info('sent data to all output IDs')
        else:
            self.logger.info('not sending null data')

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

        np.random.seed()

        # Set up the network connection:
        self.init_net()

        # Don't allow keyboard interruption of process:
        self.running = True
        with NoKeyboardInterrupt():
            while True:

                # Run the processing step:
                self.run_step()

                # Synchronize:
                if self._sync():
                    break

        self.logger.info('done')

class Broker(mp.Process):
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

        # Use the object instance's Python ID (i.e., memory address)
        # as its UID:
        self.id = str(id(self))

        self.logger = logging.getLogger('broker %s' % self.id)

        self.port_data = port_data
        self.port_ctrl = port_ctrl
        if self.port_data == self.port_ctrl:
            raise ValueError('data and control ports must differ')

        # Flag indicating when the broker instance is running:
        self.running = False

        super(Broker, self).__init__()

    def _ctrl_handler(self, msg):
        """
        Control interface handler for received data.

        Notes
        -----
        Assumes that self.ioloop, self.stream, and self.stream.ctrl
        exist.

        """

        data = msg[0]
        if data == 'quit':
            self.logger.info('recv: quit')
            self.stream.flush()
            self.stream_ctrl.flush()
            self.ioloop.stop()
            self.running = False

    def run(self):
        """
        Body of process.
        """

        self.logger.info('initializing network connection')
        self.ctx = zmq.Context()

        # Use a nonblocking port for the control interface:
        self.sock_ctrl = self.ctx.socket(zmq.DEALER)
        self.sock_ctrl.setsockopt(zmq.IDENTITY, self.id)
        self.sock_ctrl.connect("tcp://localhost:%i" % self.port_ctrl)

        # Use a nonblocking port for the data interface:
        self.sock_data = self.ctx.socket(zmq.ROUTER)
        self.sock_data.bind("tcp://*:%i" % self.port_data)

        # Set up data stream event handler:
        self.ioloop = IOLoop.instance()
        self.stream = ZMQStream(self.sock_data, self.ioloop)
        def handler(msg):

            # Don't allow interrupts to prevent the handler from
            # completely executing each time it is called:
            with NoKeyboardInterrupt():

                # The first entry of the message is the originating ID
                # (prepended by zmq); the second is the destination ID:
                data = pickle.loads(msg[2])
                #self.logger.info('recv from %s: %s' % (msg[0], data))
                #self.logger.info('sent to   %s: %s' % (msg[1], data))

                # Route to the destination ID and send the source ID
                # along with the data:
                self.sock_data.send_multipart([msg[1], msg[0], msg[2]])
        self.stream.on_recv(handler)

        # Set up control stream event handler:
        self.stream_ctrl = ZMQStream(self.sock_ctrl, self.ioloop)
        self.stream_ctrl.on_recv(self._ctrl_handler)

        # Start the event loop:
        self.ioloop.start()

        # This is reached when the event loop is shut down:
        self.logger.info('done')

class Connectivity(object):
    """
    Intermodule connectivity class.

    """

    def __init__(self):

        # Use the object instance's Python ID (i.e., memory address)
        # as its UID:
        self.id = str(id(self))

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

        # Use the object instance's Python ID (i.e., memory address)
        # as its UID:
        self.id = str(id(self))

        self.logger = logging.getLogger('manage %s' % self.id)
        self.port_data = port_data
        self.port_ctrl = port_ctrl

        # Set up a router socket to communicate with other topology components:
        self.ctx = zmq.Context()
        self.sock_ctrl = self.ctx.socket(zmq.ROUTER)
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

        with NoKeyboardInterrupt():
            for b in self.broks.values():
                b.start()
            for m in self.mods.values():
                m.start()

    def stop(self):
        """
        Stop execution of all processes.
        """

        # Tell all modules to terminate:
        for i in self.mods.keys():
            self.logger.info('sent to   %s: quit' % i)
            self.sock_ctrl.send_multipart([i, 'quit'])

        time.sleep(1)
        # Tell the brokers to terminate:
        ## Module processes should return a signal to the broker indicating
        ## when they are about to successfully terminate
        for i in self.broks.keys():
            self.logger.info('sent to   %s: quit' % i)
            self.sock_ctrl.send_multipart([i, 'quit'])


if __name__ == '__main__':

    # Log to screen and to a file:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler('exec.log', 'w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set up and start emulation:
    man = Manager()
    man.add_brok()
    m1 = man.add_mod()
    m2 = man.add_mod()
    m3 = man.add_mod()
    m4 = man.add_mod()
    conn = man.add_conn()
    man.connect(m1, m2, conn)
    man.connect(m2, m1, conn)
    #man.connect(m2, m3, conn)    
    man.connect(m3, m4, conn)
    man.connect(m4, m3, conn)
    man.connect(m4, m1, conn)
    #b = man.broks[man.broks.keys()[0]]
    man.start()
    #m= Module(net='ctrl')
    #m.start()

    time.sleep(1)
    man.stop()
