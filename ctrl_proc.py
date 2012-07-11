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

import twiggy

import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

from ctx_managers import TryExceptionOnSignal
from nk_uuid import uuid

# Use a finite linger time to prevent sockets from either hanging or
# being uncleanly terminated when shutdown:
LINGER_TIME = 2

class ControlledProcess(mp.Process):
    """
    Process subclass with control port.
    """

    def __init__(self, port_ctrl, quit_sig, *args, **kwargs):

        # Unique object identifier:
        self.id = uuid()

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
            try:
                self.stream_ctrl.flush()
                self.stream_ctrl.stop_on_recv()
                self.ioloop_ctrl.stop()
            except IOError:
                self.logger.info('streams already closed')
            except:
                self.logger.info('other error occurred')
            self.logger.info('issuing signal %s' % self.quit_sig)
            os.kill(os.getpid(), self.quit_sig)

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

    def _init_net(self, event_thread=True):
        """
        Initialize network connection.

        Parameters
        ----------
        event_thread : bool
            If True, start the control event loop in a new thread.

        """

        # Set up zmq context and event loop:
        self.ctx = zmq.Context()
        self.ioloop = IOLoop.instance()

        # Set up event loop handlers:
        self._init_ctrl_handler()

        # Start event loop:
        if event_thread:
            th.Thread(target=self.ioloop.start).start()
        else:
            self.ioloop.start()
            
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

