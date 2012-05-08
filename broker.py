#!/usr/bin/env python

"""
Broker for routing between modules run in separate processes; data is
read from all modules before data is routed between modules.
"""

import logging
import signal
import numpy as np
import multiprocessing as mp
import zmq

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
    Module to run in a process.

    Parameters
    ----------
    id : int
        Module ID.
    port : int
        Port to use when communicating with broker.

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
        self.port = kwargs.pop('port')
        self.logger = logging.getLogger('module %s' % self.id)

        mp.Process.__init__(self, *args, **kwargs)
        
    def run(self):

        # Make the module processes ignore Ctrl-C:
        orig_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Connect to the broker:
        self.ctx = zmq.Context()
        self.poller = zmq.Poller()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.IDENTITY, str(self.id))
        self.sock.connect("tcp://localhost:%i" % self.port)
        self.poller.register(self.sock, zmq.POLLIN)

        # The modules send an initialization signal after connecting:
        self.sock.send('init')

        # Wait for data to arrive:
        while True:
            data = ''
            if is_poll_in(self.sock, self.poller):
                data = self.sock.recv()
                self.logger.info('received: %s' % data)
                if data == 'quit':
                    break

                result = self.process_data(data)                
                self.sock.send(result)

        # Restore SIGINT signal handler before exiting:
        signal.signal(signal.SIGINT, orig_handler)
        self.logger.info('done')
        
    def process_data(self, data):
        """
        This method should be implemented to do something with its
        arguments and produce output.
        """

        return ''
    
class ModuleBroker(object):
    """
    Broker for communicating between modules.

    Parameters
    ----------
    port : int
        Port to use for communication with modules.

    Methods
    -------
    create(module_class)
        Create an instance of the specified module class and connect
        it to the broker.
    recv_all()
        Wait for all modules to transmit data to the broker.
    send_all(out_data)
        Send data to modules. The list must contain tuples that each
        contain the destination module ID and the data payload.
    send_all_quit()
        Send a quit signal to all modules.
    run()
        Body of broker.
    process_data(in_data)
        Process data from modules; the output should be in a format
        that can be transmitted to the modules by the `send_all()`
        method.
    
    """
    
    def __init__(self, port=5000):

        self.logger = logging.getLogger('broker  ')
        self.port = port

        # Number of modules:
        self.N = 0
        
        # Set up a router socket to communicate with the started
        # processes:
        self.ctx = zmq.Context()
        self.poller = zmq.Poller()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.bind("tcp://*:%i" % self.port)
        self.poller.register(self.sock, zmq.POLLIN)
        
        # Dictionary for mapping module IDs to module instances:
        self.id_to_mod_dict = {}

        # Dictionary for mapping module instances to module IDs:
        self.mod_to_id_dict = {}

    def create(self, module_class):
        """
        Create instance of module to the network managed by the broker.
        """

        if not issubclass(module_class, Module):
            raise ValueError('subclass of Module class must be specified')
        m = module_class(id=self.N, port=self.port)
        m.start()
        self.id_to_mod_dict[str(self.N)] = m
        self.mod_to_id_dict[m] = str(self.N)
        self.N += 1

    def recv_all(self):
        """
        Wait for data to be received from all modules.
        """
                
        # Wait for data from all of the modules:
        in_data = []
        ack_list = self.id_to_mod_dict.keys()
        while ack_list:
            if is_poll_in(self.sock, self.poller):
                from_id = self.sock.recv()
                data = self.sock.recv()
                self.logger.info('received from %s: %s' % (from_id, data))
                if from_id in ack_list:
                    ack_list.remove(from_id)

                # Save incoming data:
                in_data.append((from_id, data))

        return in_data

    def send_all(self, out_data):
        """
        Send data to all modules. Each entry in the specified list
        must be a tuple containing the destination module ID and the
        data payload.
        """

        for entry in out_data:
            self.logger.info('sent to %s: %s' % entry)
            self.sock.send_multipart(entry)

    def send_all_quit(self):
        """
        Tell all of the modules to quit.
        """
        self.send_all(zip(self.id_to_mod_dict.keys(), ['quit']*len(self.id_to_mod_dict)))
        
    def run(self):
        """
        Body of broker.
        """

        while True:

            # Get data from all modules:
            try:
                in_data = self.recv_all()
            except KeyboardInterrupt:
                break
            self.logger.info('barrier reached')
            
            # Figure out what needs to be routed to each module:
            out_data = self.process_data(in_data)
            
            # Send all data to the appropriate destinations:
            try:                
                self.send_all(out_data)
            except KeyboardInterrupt:
                break            
            
        # Tell the modules to terminate:
        self.send_all_quit()
        self.logger.info('done')
        
    def process_data(self, in_data):
        """
        Figure out how to route data entries in the specified
        list. Each entry is a tuple containing the ID of the source
        module and the data itself.
        """

        return in_data
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    b = ModuleBroker()
    b.create(Module)
    b.create(Module)
    b.create(Module)

    b.run()
