#!/usr/bin/env python

"""
Communication utilities.
"""

import zmq
import twiggy

def is_poll_in(sock, poller, timeout=100):
    """
    Check for incoming data on a socket using a poller.
    """

    socks = dict(poller.poll(timeout))
    if sock in socks and socks[sock] == zmq.POLLIN:
        return True
    else:
        return False

def get_random_port(min_port=49152, max_port=65536, max_tries=100):
    """
    Return available random ZeroMQ port.
    """

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    try:
        port = sock.bind_to_random_port('tcp://*', min_port, max_port, max_tries)
    except:
        raise zmq.ZMQError(msg='Could not find free port')
    finally:
        sock.close()
    return port

def _iterable(x):
    try:
        iter(x)
    except:
        return False
    else:
        return True

def synchronize(sock, sending, what='',
                sync_addr='ipc://sync', timeout=10):
    """
    Synchronize asynchronous ZeroMQ socket pair.

    When run on a socket pair (e.g., ROUTER/DEALER, PUB/SUB) that supports
    asynchronous communication, this function blocks until the sockets
    are in full communication by sending dummy messages from one socket to the 
    other until the latter acknowledges receiving the messages via
    a REQ/REP socket pair created on the fly.

    Parameters
    ----------
    sock : zmq.Socket
        Socket to synchronize.
    sending : bool
        Whether to send dummy messages (True) or wait for them (False).
    what : object
        What to use as a dummy message. If an iterable object is specified, 
        the message is assumed to be multipart. This can be useful for
        synchronizing a ROUTER to a DEALER with a specific identifier, for
        example.
    sync_addr : str
        Port address to use for REQ/REP socket pair. This must be the same
        in both invocations of the function.
    timeout : int
        Timeout for poller used to detect synchronization.

    Notes
    -----
    Only works for socket pairs, i.e., cannot be used to synchronize multiple SUB
    sockets with a single PUB socket.
    """

    # XXX should use six.string_types to ensure Python 3 compatibility XXX
    if _iterable(what) and not isinstance(what, basestring):
        send_func = sock.send_multipart
        recv_func = sock.recv_multipart
    else:
        send_func = sock.send
        recv_func = sock.recv

    if sending:
        # Send dummy messages until sync is detected by remote:
        sock_sync = sock.context.socket(zmq.REP)
        sock_sync.connect(sync_addr)
        while True:
            send_func(what)

            # Stop transmitting dummy messages when sync is acknowledged:
            if sock_sync.poll(timeout):
                sock_sync.recv()
                sock_sync.send('ack')
                break
        sock_sync.close()
    else:
        sock_sync = sock.context.socket(zmq.REQ)
        sock_sync.bind(sync_addr)

        # Wait for dummy messages to start flowing:
        while True:
            if sock.poll(timeout):
                recv_func()

                # When dummy messages are detected, acknowledge the sync
                # and exit:
                sock_sync.send('')
                sock_sync.recv()
                break
        sock_sync.close()
    
class ZMQOutput(twiggy.outputs.Output):
    """
    Output messages to a ZeroMQ PUB socket.
    """
    
    def __init__(self, addr, mode,
                 format=None, close_atexit=True):
        self.addr = addr
        self.mode = mode
        super(ZMQOutput, self).__init__(format, close_atexit)
        
    def _open(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        if self.mode == 'bind':
            self.sock.bind(addr)
        elif self.mode == 'connect':
            self.sock.connect(addr)
        else:
            raise ValueError('invalid connection mode')

    def _close(self):
        self.sock.close()

    def _write(self, x):
        self.sock.send(x)

