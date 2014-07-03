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

import signal, sys, time
import multiprocessing as mp
import threading as th

import twiggy

import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

# Use a finite linger time to prevent sockets from either hanging or
# being uncleanly terminated when shutdown:
LINGER_TIME = 2

class ControlledProcess(mp.Process):
    """
    Process subclass with control port.

    Parameters
    ----------
    port_ctrl : int
        Network port for receiving control messages.
    quit_sig : int
        OS signal to use when quitting the proess.
    id : str
        Unique object identifier. Used for communication and logging.

    See Also
    --------
    multiprocessing.Process
    """

    def __init__(self, port_ctrl, id, *args, **kwargs):

        # Unique object identifier:
        self.id = id

        # Logging:
        self.logger = twiggy.log.name(self.id)

        # Control port:
        self.port_ctrl = port_ctrl

        # Flag to use when stopping the process:
        self.running = False

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
            except Exception as e:
                self.logger.info('other error occurred: '+e.message)
            self.running = False

    def _init_ctrl_handler(self):
        """
        Initialize control port handler.
        """

        # Set the linger period to prevent hanging on unsent messages
        # when shutting down:
        self.logger.info('initializing ctrl handler')
        self.sock_ctrl = self.zmq_ctx.socket(zmq.DEALER)
        self.sock_ctrl.setsockopt(zmq.IDENTITY, self.id)
        self.sock_ctrl.setsockopt(zmq.LINGER, LINGER_TIME)
        self.sock_ctrl.connect('tcp://localhost:%i' % self.port_ctrl)

        self.stream_ctrl = ZMQStream(self.sock_ctrl, self.ioloop_ctrl)
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
        self.zmq_ctx = zmq.Context()
        self.ioloop_ctrl = IOLoop.instance()

        # Set up event loop handlers:
        self._init_ctrl_handler()

        # Start event loop:
        if event_thread:
            th.Thread(target=self.ioloop_ctrl.start).start()
        else:
            self.ioloop_ctrl.start()

    def run(self):
        """
        Body of process.
        """

        self._init_net()
        self.running = True
        while True:
            self.logger.info('idling')
            if not self.running:
                self.logger.info('stopping run loop')
                break
        self.logger.info('done')

if __name__ == '__main__':
    output = twiggy.outputs.StreamOutput(twiggy.formats.line_format,
                                         stream=sys.stdout)
    twiggy.emitters['*'] = twiggy.filters.Emitter(twiggy.levels.DEBUG,
                                                  True, output)

    zmq_ctx = zmq.Context()
    sock = zmq_ctx.socket(zmq.ROUTER)
    port_ctrl = sock.bind_to_random_port('tcp://*')

    # Protect both the child and parent processes from being clobbered by
    # Ctrl-C:
    from ctx_managers import TryExceptionOnSignal, IgnoreKeyboardInterrupt
    with IgnoreKeyboardInterrupt():
        p = ControlledProcess(port_ctrl, 'mymod')
        p.start()

        time.sleep(3)
        sock.send_multipart([p.id, 'quit'])
