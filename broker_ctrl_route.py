#!/usr/bin/env python

"""
Broker for routing between modules run in separate processes; data is
read from all modules before data is routed between modules.
Uses a separate port for control of modules.
"""

import logging, time
import multiprocessing as mp
import cPickle as pickle
import numpy as np
import zmq
from zmq.eventloop.ioloop import IOLoop, PeriodicCallback
from zmq.eventloop.zmqstream import ZMQStream

from int_ctx import NoKeyboardInterrupt, OnKeyboardInterrupt

log_format = lambda s: "%-15s" % s

class Module(mp.Process):
    """
    Module to run in a process.

    Parameters
    ----------
    id : int
        Module ID.
    port_data : int
        Port to use when communicating with broker.
    port_ctrl : int
        Port used by broker to control module.

    Methods
    -------
    run()
        Body of process.
    process_data(data)
        Processes the specified data and returns a result for
        transmission to other modules.

    """

    def __init__(self, *args, **kwargs):
        self.id = kwargs.pop('id')
        self.port_data = kwargs.pop('port_data')
        self.port_ctrl = kwargs.pop('port_ctrl')
        self.logger = logging.getLogger('module %-2s' % self.id)

        mp.Process.__init__(self, *args, **kwargs)

    def run(self):

        # Don't allow interrupts to prevent the handler from
        # completely executing each time it is called:                             
        with NoKeyboardInterrupt():

            # Use a blocking port for the data interface to ensure that
            # each step of the simulation waits for transmission of data
            # to complete:
            self.ctx = zmq.Context()
            self.sock_data = self.ctx.socket(zmq.REQ)
            self.sock_data.setsockopt(zmq.IDENTITY, str(self.id))
            self.sock_data.connect("tcp://localhost:%i" % self.port_data)

            # Use a nonblocking port for the control interface:
            self.sock_ctrl = self.ctx.socket(zmq.DEALER)
            self.sock_ctrl.setsockopt(zmq.IDENTITY, str(self.id))
            self.sock_ctrl.connect("tcp://localhost:%i" % self.port_ctrl)

            # Set up event loop:
            self.ioloop = IOLoop.instance()
            self.stream_ctrl = ZMQStream(self.sock_ctrl, self.ioloop)

            # Run the processing step and the data transmission:
            np.random.seed()
            def step():
                result = self.process_data(step.data)
                self.sock_data.send(result)
                self.logger.info(log_format('sent:')+result)
                data = self.sock_data.recv()
                data = pickle.loads(data)
                self.logger.info(log_format('received:')+str(data))
            step.data = ''
            pc = PeriodicCallback(step, 1, self.ioloop)

            # Handle quit signals:
            def handler(msg):
                data = msg[0]
                if data == 'quit':
                    self.logger.info('quit received')
                    self.stream_ctrl.flush()
                    pc.stop()
                    self.ioloop.stop()
            self.stream_ctrl.on_recv(handler)

            # Start the processing:
            pc.start()
            self.ioloop.start()

        self.logger.info('done')

    def process_data(self, data):
        """
        This method should be implemented to do something with its
        arguments and produce output.
        """

        return str(np.random.randint(0, 10))

class ModuleBroker(object):
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
    create(module_class)
        Create an instance of the specified module class and connect
        it to the broker.
    run()
        Body of broker.
    process_data(in_data)
        Route data entries.

    """

    def __init__(self, port_data=5000, port_ctrl=5001):

        self.logger = logging.getLogger('broker   ')
        self.port_data = port_data
        self.port_ctrl = port_ctrl

        # Set up a router socket to communicate with the started
        # processes:
        self.ctx = zmq.Context()
        self.sock_data = self.ctx.socket(zmq.ROUTER)
        self.sock_data.bind("tcp://*:%i" % self.port_data)

        self.sock_ctrl = self.ctx.socket(zmq.ROUTER)
        self.sock_ctrl.bind("tcp://*:%i" % self.port_ctrl)

        # Dictionary for mapping module IDs to module instances:
        self.id_to_mod_dict = {}

        # Dictionary for mapping module instances to module IDs:
        self.mod_to_id_dict = {}

        # Intermodule connectivity matrix:
        self.route_table = np.empty((), int)

    @property
    def N(self):
        """
        Number of module instances.
        """

        return len(self.id_to_mod_dict)

    def create(self, module_class):
        """
        Create instance of module to the network managed by the broker.
        """

        if not issubclass(module_class, Module):
            raise ValueError('subclass of Module class must be specified')
        m = module_class(id=self.N, port_data=self.port_data, port_ctrl=self.port_ctrl)
        m.start()
        self.id_to_mod_dict[str(self.N)] = m
        self.mod_to_id_dict[m] = str(self.N)

    def set_route_table(self, table):
        """
        Set intermodular connectivity matrix.
        """

        s = np.shape(table)
        if len(s) != 2 or len(np.unique(s)) != 1 or s[0] != self.N:
            raise ValueError('invalid routing table')
        self.route_table = np.copy(table)

    def run(self):
        """
        Body of broker.
        """

        self.ioloop = IOLoop.instance()
        self.stream = ZMQStream(self.sock_data, self.ioloop)
        def handler(msg):

            # Don't allow interrupts to prevent the handler from
            # completely executing each time it is called:                                         
            with NoKeyboardInterrupt():
                self.logger.info(log_format('recv from %s:' % msg[0])+str(msg[2]))

                # Don't process anything until data is received from
                # all of the modules:
                if msg[0] in handler.recv_list:
                    handler.in_data.append(msg)
                    handler.recv_list.remove(msg[0])

                if len(handler.recv_list) == 0:
                    self.logger.info('all data received')
                    out_data = self.process_data(handler.in_data)

                    # Serialize each output entry's data list before transmission:
                    for entry in out_data:
                        self.logger.info(log_format('sent to %s:' % entry[0])+str(entry[2]))
                        self.sock_data.send_multipart([entry[0],
                                                       entry[1], pickle.dumps(entry[2])])

                    # Reset list of modules from which data was
                    # received and list of received data:
                    handler.in_data = []                                        
                    handler.recv_list = self.id_to_mod_dict.keys()

        handler.in_data = []
        handler.recv_list = self.id_to_mod_dict.keys()                
        self.stream.on_recv(handler)
        def on_interrupt(signum, frame):

            # Need to wait for a little while for modules to stop:
            time.sleep(1)
            self.stream.flush()
            self.ioloop.stop()

        with OnKeyboardInterrupt(on_interrupt):
            self.ioloop.start()

        # Tell the modules to terminate:
        for i in self.id_to_mod_dict.keys():
            entry = (str(i), 'quit')
            self.logger.info(log_format('sent to %s:' % entry[0])+entry[1])
            self.sock_ctrl.send_multipart(entry)
        self.logger.info('done')

    def process_data(self, in_data):
        """
        Route data entries.
        
        Figure out how to route data entries in the specified
        list. Each entry is a tuple containing the ID of the source
        module and the data itself.

        Parameters
        ----------
        in_data : list of tuples
            Input data to route. Each tuple in this list contains (source address, '',
            data).

        Returns
        -------
        out_data : list of tuples
            Routed data to transmit. Each tuple in this list contains (destination address, '',
            [data list]). 
            
        Notes
        -----
        The routing table is assumed to map source modules (row
        indices) to destination modules (column indices) if the table entry
        at (row, col) is nonzero.

        """

        # Create a destination entry for each destination module:
        out_data = [(str(i), '', []) for i in xrange(self.route_table.shape[1])]

        # Append the data in each source entry to the appropriate
        # destination entry's data list:
        for entry in in_data:
            src, _, data = entry
            for dest in np.nonzero(self.route_table[src, :])[0]:
                out_data[dest][2].append(data)
        return out_data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    b = ModuleBroker()
    N = 50
    for i in xrange(N):
        b.create(Module)
    b.set_route_table(np.random.randint(0, 2, (b.N, b.N)))
    b.run()
