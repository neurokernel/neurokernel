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

import mixins

# Use a finite linger time to prevent sockets from either hanging or
# being uncleanly terminated when shutdown:
LINGER_TIME = 2

class ControlledProcess(mixins.LoggerMixin, mp.Process):
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
        mixins.LoggerMixin.__init__(self, id)

        # Control port:
        self.port_ctrl = port_ctrl

        # Flag to use when stopping the process:
        self.running = False

        mp.Process.__init__(self, *args, **kwargs)

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
    from neurokernel.tools.comm import sync_pub, sync_sub

    output = twiggy.outputs.StreamOutput(twiggy.formats.line_format,
                                         stream=sys.stdout)
    twiggy.emitters['*'] = twiggy.filters.Emitter(twiggy.levels.DEBUG,
                                                  True, output)

    # Example: create one custom controlled process that emits data on a port,
    # and another that 
    class MyControlledProcess(ControlledProcess):
        def run(self):
            """
            Body of process.
            """

            self._init_net()
            sock_out = self.zmq_ctx.socket(zmq.PUB)
            sock_out.bind('ipc://out')
            sync_pub(sock_out, ['listen'])

            self.running = True
            counter = 0
            while True:
                sock_out.send(str(counter))
                counter += 1
                self.logger.info('sent %s' % counter)
                if not self.running:
                    self.logger.info('stopping run loop')
                    break
            self.logger.info('done')

    class MyListenerProcess(ControlledProcess):
        def run(self):
            """
            Body of process.
            """

            self._init_net()
            sock_out = self.zmq_ctx.socket(zmq.SUB)
            sock_out.setsockopt(zmq.SUBSCRIBE, '')
            sock_out.connect('ipc://out')
            sync_sub(sock_out, 'listen')

            self.running = True
            while True:
                if sock_out.poll(10):
                    data = sock_out.recv()
                    self.logger.info('received %s' % data)
                if not self.running:
                    self.logger.info('stopping run loop')
                    break
            self.logger.info('done')

    # Sockets for controlling started processes:
    zmq_ctx = zmq.Context()
    sock_myproc = zmq_ctx.socket(zmq.ROUTER)
    sock_mylist = zmq_ctx.socket(zmq.ROUTER)
    port_myproc = sock_myproc.bind_to_random_port('tcp://*')
    port_mylist = sock_mylist.bind_to_random_port('tcp://*')

    # Protect both the child and parent processes from being clobbered by
    # Ctrl-C:
    from ctx_managers import TryExceptionOnSignal, IgnoreKeyboardInterrupt
    with IgnoreKeyboardInterrupt():
        myproc = MyControlledProcess(port_myproc, 'myproc')
        myproc.start()
        mylist = MyListenerProcess(port_mylist, 'mylist')
        mylist.start()
        time.sleep(1)
        sock_myproc.send_multipart([myproc.id, 'quit'])
        sock_mylist.send_multipart([mylist.id, 'quit'])
