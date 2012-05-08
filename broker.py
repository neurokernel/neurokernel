#!/usr/bin/env python

"""
Broker for routing between modules run in separate processes; data is
read from all modules before data is routed between modules.
"""

import time
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
    broker_port : int
        Port to use when communicating with other modules through
        broker.

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
        self.broker_port = kwargs.pop('broker_port')

        mp.Process.__init__(self, *args, **kwargs)
        
    def run(self):

        # Connect to the broker:
        self.ctx = zmq.Context()
        self.poller = zmq.Poller()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.IDENTITY, str(self.id))
        self.sock.connect("tcp://localhost:%i" % self.broker_port)
        self.poller.register(self.sock, zmq.POLLIN)

        # The modules send an initialization signal after connecting:
        self.sock.send('init')

        # Wait for data to arrive:
        while True:
            data = ''
            if is_poll_in(self.sock, self.poller):
                data = self.sock.recv()
                print 'process %i <- %s' % (self.id, data)
                if data == 'quit':
                    break

                result = self.process_data(data)                
                self.sock.send(result)

    def process_data(self, data):
        """
        This method should be implemented to do something with its
        arguments and produce output.
        """

        return ''
    
class ModuleBroker(object):
    
    def __init__(self, N, broker_port=5000):
        self.N = N
        self.broker_port = broker_port
        
        # Set up a router socket to communicate with the started
        # processes:
        self.ctx = zmq.Context()
        self.poller = zmq.Poller()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.bind("tcp://*:%i" % self.broker_port)
        self.poller.register(self.sock, zmq.POLLIN)
        
        # Create and start all of the processes:
        self.proc_list = []
        for i in xrange(N):
            m = Module(id=i, broker_port=5000)
            m.start()
            self.proc_list.append(m)

        # Allow modules time to start:
        time.sleep(1)

    def run(self):
        """
        Route data between modules.
        """

        for i in xrange(10):
            
            # Wait for data from all of the modules:
            in_data = []
            ack_list = range(self.N)
            while ack_list:
                if is_poll_in(self.sock, self.poller):
                    from_id = self.sock.recv()
                    data = self.sock.recv()
                    print 'broker <- %s: %s' % (from_id, data)
                    ack_list.remove(int(from_id))

                    # Save incoming data:
                    in_data.append((from_id, data))
            print 'broker: barrier reached'
             
            # Figure out what needs to be routed to each module:
            out_data = self.route(in_data)
            
            # Send all data to the appropriate destinations:
            for entry in out_data:
                print 'broker -> %s: %s' % entry
                self.sock.send_multipart(entry)

        # Tell the modules to terminate:
        for i in xrange(self.N):
            self.sock.send_multipart((str(i), 'quit'))

    def route(self, in_data):
        """
        Figure out how to route data entries in the specified
        list. Each entry is a tuple containing the ID of the source
        module and the data itself.
        """

        return in_data
    
if __name__ == '__main__':
    b = ModuleBroker(3)

    b.run()
