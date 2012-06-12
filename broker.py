#!/usr/bin/env python

"""
Broker for routing between modules run in separate processes; data is
read from all modules before data is routed between modules.
"""

import logging, signal
import multiprocessing as mp
import numpy as np
import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

from int_ctx import NoKeyboardInterrupt, OnKeyboardInterrupt
        
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
        with NoKeyboardInterrupt():

            # Connect to the broker:
            self.ctx = zmq.Context()
            self.sock = self.ctx.socket(zmq.DEALER)
            self.sock.setsockopt(zmq.IDENTITY, str(self.id))
            self.sock.connect("tcp://localhost:%i" % self.port)

            # The modules send an initialization signal after connecting:
            self.sock.send('init')

            # Wait for data to arrive:
            self.ioloop = IOLoop.instance()
            self.stream = ZMQStream(self.sock, self.ioloop)
            def handler(msg):
                data = msg[0].decode()
                self.logger.info('received: %s' % data)
                if data == 'quit':
                    self.stream.flush()
                    self.ioloop.stop()
                result = self.process_data(data)
                self.sock.send(result)
            self.stream.on_recv(handler)
            self.ioloop.start()
        
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

        # Set up a router socket to communicate with the started
        # processes:
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.bind("tcp://*:%i" % self.port)
        
        # Dictionary for mapping module IDs to module instances:
        self.id_to_mod_dict = {}

        # Dictionary for mapping module instances to module IDs:
        self.mod_to_id_dict = {}

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
        m = module_class(id=self.N, port=self.port)
        m.start()
        self.id_to_mod_dict[str(self.N)] = m
        self.mod_to_id_dict[m] = str(self.N)
                    
    def run(self):
        """
        Body of broker.
        """

        self.ioloop = IOLoop.instance()
        self.stream = ZMQStream(self.sock, self.ioloop)
        def handler(msg):

            # Stop the event loop when an interrupt occurs:
            def on_interrupt(signum, frame):
                self.stream.flush()
                self.ioloop.stop()
            with OnKeyboardInterrupt(on_interrupt):
                
                # Need to cast the message contents to non-Unicode
                # strings for some reason:
                addr = str(msg[0].decode())
                data = str(msg[1].decode())
                self.logger.info('received from %s: %s' % (addr, data))
                if addr in handler.ack_list:
                    handler.ack_list.remove(addr)
                handler.in_data.append((addr, data))
                if len(handler.ack_list) == 0:
                    self.logger.info('barrier reached')
                    out_data = self.process_data(handler.in_data)          
                    for entry in out_data:
                        self.logger.info('sent to %s: %s' % entry)
                        self.sock.send_multipart(entry)
                            
                    # Reset variables:
                    handler.ack_list = self.id_to_mod_dict.keys()
                    handler.in_data = []        
                            
        handler.ack_list = self.id_to_mod_dict.keys()
        handler.in_data = []
        self.stream.on_recv(handler)
        with OnKeyboardInterrupt(lambda signum, frame: self.ioloop.stop()):
            self.ioloop.start()
            
        # Tell the modules to terminate:
        for i in self.id_to_mod_dict.keys():
            entry = (str(i), 'quit')
            self.logger.info('sent to %s: %s' % entry)
            self.sock.send_multipart(entry)
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
    N = 3
    for i in xrange(N):
        b.create(Module)
    b.run()
