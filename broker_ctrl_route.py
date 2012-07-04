#!/usr/bin/env python

"""
Broker for routing between modules run in separate processes; data is
read from all modules before data is routed between modules.
Uses a separate port for control of modules.
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

log_format = lambda s: "%-15s" % s

PORT_DATA = 5000
PORT_CTRL = 5001

class Module(mp.Process):
    """
    Processing module.

    This class repeatedly executes a work method until it receives a
    termination signal.

    Parameters
    ----------
    net: str
        Network connectivity. May be `unconnected` for no connection,
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
        if value not in ['unconnected', 'in', 'out', 'full']:
            raise ValueError('invalid network connectivity value')
        self.logger.info('net status changed: %s -> %s' % (self._net, value))
        self._net = value

    def __init__(self, net='unconnected',
                 port_data=PORT_DATA, port_ctrl=PORT_CTRL):

        # Use the object instance's Python ID (i.e., memory address)
        # as its UID:
        # NOTE: since object instance IDs are only unique for
        # instantiated objects, generated unique identifiers should
        # eventually used if the dynamic creation and destruction of
        # modules must eventually be supported:
        self.id = str(id(self))

        self.logger = logging.getLogger('module %s' % self.id)

        self.net = net
        self.port_data = port_data
        self.port_ctrl = port_ctrl
        if self.port_data == self.port_ctrl:
            raise ValueError('data and control ports must differ')

        # Flag indicating when the module instance is running:
        self.running = False

        # Attributes used for input and output:
        self.in_data = None
        self.out_data = None

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
            self.logger.info('quit received')
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

                ### XXX replacing with nonblocking port:
                # Use a blocking port for the data interface to ensure that
                # each step of the simulation waits for transmission of data
                # to complete:
                self.sock_data = self.ctx.socket(zmq.DEALER)
                self.sock_data.setsockopt(zmq.IDENTITY, self.id)
                self.sock_data.connect("tcp://localhost:%i" % self.port_data)

                # Set up control stream event handler:
                self.ioloop = IOLoop.instance()
                self.stream = ZMQStream(self.sock_ctrl, self.ioloop)
                self.stream.on_recv(self._ctrl_handler)

                # Start the event loop:
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

        """

        if self.net == 'unconnected':
            self.logger.info('not synchronizing with network')
        else:
            self.logger.info('synchronizing with network')
            if self.net in ['out', 'full']:
                self.sock_data.send(pickle.dumps(self.out_data))
                self.logger.info(log_format('sent:')+self.out_data)
            if self.net in ['in', 'full']:
                self.in_data = pickle.loads(self.sock_data.recv())
                self.logger.info(log_format('received:')+str(self.in_data))

    def run_step(self):
        """
        This method should be implemented to do something interesting.

        Notes
        -----
        Assumes that the attributes used for input and output already
        exist.

        """

        self.logger.info('running execution step')
        self.out_data = str(np.random.randint(0, 10))

    def run(self):
        """
        Body of process.
        """

        # Set up the network connection:
        self.init_net()

        # Don't allow keyboard interruption of process:
        self.running = True
        with NoKeyboardInterrupt():

            # XXX remove the following line:
            np.random.seed()
            while self.running:

                time.sleep(1)
                # Run the processing step:
                self.run_step()

                # Synchronize:
                self._sync()

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
    routing_table : RoutingTable
        Table describing connections between modules.

    Methods
    -------
    run()
        Body of process.
    route()
        Route data entries.
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

        # Intermodule connectivity:
        self.routing_table = routing_table

        # List of module identifiers (assumes that the routing table's
        # row and column identifiers are identical):
        self.mod_ids = routing_table.ids

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
            self.logger.info('quit received')
            self.stream.flush()
            self.stream_ctrl.flush()
            self.ioloop.stop()
            self.running = False

    def run(self):
        """
        Body of process.
        """

        # Find the modules in the routing table that expect input
        # (some might only produce data and therefore not require any
        # input from the network):
        ids = self.routing_table.ids
        recv_list = [ids[i] for i, e in \
                     enumerate([sum(self.routing_table[:, k]) for k in ids]) if e != 0]

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
                self.logger.info(log_format('recv from %s: ' % msg[0])+str(msg[1]))
                self.logger.info('recv_list before: '+str(handler.recv_list))

                # Wait for data to be received from all of the
                # modules expecting input from other modules:
                if msg[0] in handler.recv_list:
                    handler.in_data[msg[0]] = msg[1]
                    handler.recv_list.remove(msg[0])
                    self.logger.info('recv_list after: '+str(handler.recv_list))
                if len(handler.recv_list) == 0:
                    self.logger.info('all data received')
                    out_data = self.route(handler.in_data)

                    # Serialize each output entry's data list before transmission:
                    for dest_id in out_data.keys():
                        self.logger.info(log_format('sent to %s: ' % dest_id)+str(out_data[dest_id]))
                        self.sock_data.send_multipart([dest_id, pickle.dumps(out_data[dest_id])])
                    self.logger.info('all data sent')

                    # Reset list of modules from which data was
                    # received and dict of received data:
                    handler.in_data = {}
                    handler.recv_list = copy.copy(recv_list)

        handler.in_data = {}
        handler.recv_list = copy.copy(recv_list)
        self.stream.on_recv(handler)

        # Set up control stream event handler:
        self.stream_ctrl = ZMQStream(self.sock_ctrl, self.ioloop)
        self.stream_ctrl.on_recv(self._ctrl_handler)

        # Start the event loop:
        #threading.Thread(target=self.ioloop.start).start()
        self.ioloop.start()
        self.logger.info('done')
        
    def route(self, in_data):
        """
        Route data entries.

        Figure out how to route data entries in the specified
        list. Each entry is a tuple containing the ID of the source
        module and the data itself.

        Parameters
        ----------
        in_data : dict
            Input data to route. Maps source ID (key) to data (value).

        Returns
        -------
        out_data : list of tuples
            Routed data to transmit. Maps destination ID (key) to list
            of data (value).

        Notes
        -----
        The routing table is assumed to map source modules (row
        IDs) to destination modules (column IDs) if the table entry
        at (row, col) is nonzero.

        """

        # Create a destination entry for each destination module:
        out_data = {i:[] for i in self.routing_table.ids}

        # Append the data in each source entry to the appropriate
        # destination entry's data list:
        for src in in_data.keys():

            # Find the IDs of destination modules to which the source
            # module is connected:
            dest_ids = [self.routing_table[src, :].label[0][i] for i, e in \
                        enumerate(self.routing_table[src, :]) if e != 0]
            for dest in dest_ids:
                out_data[dest].append(in_data[src])
        return out_data

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
        
        self.logger = logging.getLogger('manager')
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
        self.routing_table[m_dest.id, m_src.id] = 1

        # Update the routing table of the broker instances:
        for b in self.broks.values():
            b.routing_table[m_dest.id, m_src.id] = 1

        # Update the network connectivity of the source and
        # destination module instances if necessary:
        if m_src.net == 'unconnected':
            m_src.net = 'out'
        if m_src.net == 'in':
            m_src.net = 'full'
        if m_dest.net == 'unconnected':
            m_dest.net = 'in'
        if m_dest.net == 'out':
            m_dest = 'full'

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

        # Send quit signal when keyboard interrupts are caught:
        def on_interrupt(signum, frame):

            # Tell the modules to terminate:
            for i in self.mods.keys():
                entry = (i, 'quit')
                self.logger.info(log_format('sent to module %s:' % entry[0])+entry[1])
                self.sock_ctrl.send_multipart(entry)

            # Tell the brokers to terminate:
            for i in self.broks.keys():
                entry = (i, 'quit')
                self.logger.info(log_format('sent to broker %s:' % entry[0])+entry[1])
                self.sock_ctrl.send_multipart(entry)

        with OnKeyboardInterrupt(on_interrupt):
            for b in self.broks.values():
                b.start()
            for m in self.mods.values():
                m.start()

if __name__ == '__main__':

    # Log to screen and to a file:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler('exec.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set up and start emulation:
    man = Manager()
    man.add_brok()
    m1 = man.add_mod()
    m2 = man.add_mod()
    conn = man.add_conn()
    man.connect(m1, m2, conn)
    b = man.broks[man.broks.keys()[0]]
    man.start()
