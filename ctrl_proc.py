#!/usr/bin/env python

"""
Controlled Process subclass.
"""

import logging, os, signal, time
import multiprocessing as mp
import threading as th
from contextlib import contextmanager

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
        self.logger = logging.getLogger(self.id)        

        # Control port:
        self.port_ctrl = port_ctrl

        # Signal to use when quitting:
        self.quit_sig = quit_sig
        super(ControlledProcess, self).__init__(*args, **kwargs)

    def _ctrl_handler(self, msg):
        """
        Handle messages received by control port.
        """
        
        self.logger.info('recv: %s' % str(msg))
        if msg[0] == 'quit':
            self.stream_ctrl.flush()
            self.ioloop_ctrl.stop()
            self.logger.info('issuing signal %s' % self.quit_sig)
            os.kill(os.getpid(), self.quit_sig)
        
    def _init_ctrl_handler(self):
        """
        Initialize control port handler.
        """
        
        self.ctx = zmq.Context()
        self.sock_ctrl = self.ctx.socket(zmq.DEALER)
        self.sock_ctrl.setsockopt(zmq.IDENTITY, self.id)
        self.sock_ctrl.connect('tcp://localhost:%i' % self.port_ctrl)

        self.ioloop_ctrl = IOLoop.instance()
        self.stream_ctrl = ZMQStream(self.sock_ctrl, self.ioloop_ctrl)
        self.stream_ctrl.on_recv(self._ctrl_handler)
        th.Thread(target=self.ioloop_ctrl.start).start()

    def run(self):
        with TryExceptionOnSignal(self.quit_sig):
            self._init_ctrl_handler()
            while True:
                print 'idling'
                # Calling the logging module may cause problems! See
                # http://stackoverflow.com/questions/4601674/signal-handlers-and-logging-in-python
                #self.logger.info('idling')
        self.logger.info('exiting')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    PORT_CTRL = 6001
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind('tcp://*:%i' % PORT_CTRL)

    p = ControlledProcess(PORT_CTRL, signal.SIGUSR1)
    p.start()
    time.sleep(3)
    sock.send_multipart([p.id, 'quit'])
        
