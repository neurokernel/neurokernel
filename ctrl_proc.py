#!/usr/bin/env python

"""
Controlled Process subclass.

Notes
-----
This module uses twiggy for logging because calling the logging module
from code that is controlled by a signal handler may cause problems
[1].

.. [1] http://stackoverflow.com/questions/4601674/signal-handlers-and-logging-in-python

"""

import os, signal, sys, time
import multiprocessing as mp
import threading as th
from contextlib import contextmanager

import twiggy

import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

@contextmanager
def ExceptionOnSignal(s=signal.SIGUSR1, e=Exception):
    """
    Raise a specific exception when the specified signal is detected.
    """

    def handler(signum, frame):
        raise e('signal %i detected' % s)
    signal.signal(s, handler)
    yield

@contextmanager
def TryExceptionOnSignal(s=signal.SIGUSR1, e=Exception):
    """
    Check for exception raised in response to specific signal.
    """
    
    with ExceptionOnSignal(s, e):
        try:
            yield
        except e:
            pass
    
class ControlledProcess(mp.Process):
    """
    Process subclass with control port.
    """
    
    def __init__(self, port_ctrl, quit_sig, *args, **kwargs):

        # Use the object instance's Python ID (i.e., memory address)
        # as its UID:
        self.id = str(id(self))

        # Logging:
        self.logger = twiggy.log.name(self.id)        

        # Control port:
        self.port_ctrl = port_ctrl

        # Signal to use when quitting:
        self.quit_sig = quit_sig
        super(ControlledProcess, self).__init__(*args, **kwargs)
        
    def _ctrl_handler(self, msg):
        """
        Control port handler.
        """
        
        self.logger.info('recv: %s' % str(msg))
        if msg[0] == 'quit':
            self.stream_ctrl.flush()
            self.logger.info('issuing signal %s' % self.quit_sig)
            os.kill(os.getpid(), self.quit_sig)
            
    def _init_ctrl_handler(self):
        """
        Initialize control port handler.
        """

        # Set the linger period to 0 to prevent hanging on unsent
        # messages when shutting down:
        self.sock_ctrl = self.ctx.socket(zmq.DEALER)
        self.sock_ctrl.setsockopt(zmq.IDENTITY, self.id)
        self.sock_ctrl.setsockopt(zmq.LINGER, 0)
        self.sock_ctrl.connect('tcp://localhost:%i' % self.port_ctrl)

        self.ioloop_ctrl = IOLoop.instance()
        self.stream_ctrl = ZMQStream(self.sock_ctrl, self.ioloop_ctrl)
        self.stream_ctrl.on_recv(self._ctrl_handler)
        th.Thread(target=self.ioloop_ctrl.start).start()

    def _init_net(self):
        """
        Initialize network connection.
        """
        
        self.ctx = zmq.Context()
        self._init_ctrl_handler()
                
    def run(self):
        """
        Body of process.
        """
        
        with TryExceptionOnSignal(self.quit_sig):
            self._init_net()
            while True:
                self.logger.info('idling')
        self.logger.info('exiting')

if __name__ == '__main__':
    output = twiggy.outputs.StreamOutput(twiggy.formats.line_format,
                                         stream=sys.stdout)
    twiggy.emitters['*'] = twiggy.filters.Emitter(twiggy.levels.DEBUG,
                                                  True, output)
    
    PORT_CTRL = 6001
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind('tcp://*:%i' % PORT_CTRL)

    p = ControlledProcess(PORT_CTRL, signal.SIGUSR1)
    p.start()
    time.sleep(3)
    sock.send_multipart([p.id, 'quit'])
        
