#!/usr/bin/env python

"""
Communication utilities.
"""

import numbers

import twiggy
import zmq

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

def sync_pub(sock, ids, sync_addr='ipc://sync', timeout=10):
    """
    Synchronize a single PUB socket with multiple SUB sockets.
    Must be paired with a call to `sync_sub`.

    Parameters
    ----------
    sock : zmq.Socket
        PUB socket to synchronize.
    ids : sequence of str or int
        If a sequence, specifies the IDs associated with SUB sockets. If an 
        integer, specifies the number of SUB sockets to synchronize.
    sync_addr : str
        Port address to use for synchronization socket pair. This 
        must be the same as that passed to `sync_sub`.
    timeout : int
        Polling timeout.

    Notes
    -----
    Some synchronization messages may be delivered to the SUB sockets
    after this routine exits.
    """

    assert sock.getsockopt(zmq.TYPE) == zmq.PUB
    sock_sync = sock.context.socket(zmq.ROUTER)
    sock_sync.bind(sync_addr)    
    if isinstance(ids, numbers.Integral):

        # ids indicates the number of subscribers:
        id_set = set()
        while True:
            sock.send('')

            if sock_sync.poll(timeout):
                id, _ = sock_sync.recv_multipart()
                id_set.add(id)
            if len(id_set) == ids:
                break
    else:
        
        # ids contains the subscriber sync port IDs:
        id_set = set(ids)
        while True:
            sock.send('')

            if sock_sync.poll(timeout):
                id, _ = sock_sync.recv_multipart()
                try:
                    id_set.remove(id)
                except:
                    pass
            if not id_set:
                break

def sync_sub(sock, id, sync_addr='ipc://sync', timeout=10):
    """
    Synchronize a SUB socket with a single PUB socket.
    Must be paired with a call to `sync_pub`.

    Parameters
    ----------
    sock : zmq.Socket
        SUB socket to synchronize.
    id : str
        IDs associated with SUB socket.
    sync_addr : str
        Port address to use for synchronization socket pair. This 
        must be the same as that passed to `sync_pub`.
    timeout : int
        Polling timeout.

    Notes
    -----
    Some synchronization messages may be delivered to the specified SUB socket
    after this routine exits.
    """

    assert sock.getsockopt(zmq.TYPE) == zmq.SUB
    sock_sync = sock.context.socket(zmq.DEALER)
    sock_sync.setsockopt(zmq.IDENTITY, id)
    sock_sync.connect(sync_addr)

    while True:
        if sock.poll(timeout):
            sock.recv()
            sock_sync.send('')
            break

def sync_router(sock, ids, sync_addr='ipc://sync', timeout=10):
    """
    Synchronize a single ROUTER socket with multiple DEALER sockets.
    Must be paired with a call to `sync_dealer`.

    Parameters
    ----------
    sock : zmq.Socket
        ROUTER socket to synchronize.
    ids : sequence of str
        IDs associated with DEALER synchronization sockets
    sync_addr : str
        Port address to use for synchronization socket pair. This 
        must be the same as that passed to `sync_dealer`.
    timeout : int
        Polling timeout.
    """

    assert sock.getsockopt(zmq.TYPE) == zmq.ROUTER
    sock_sync = sock.context.socket(zmq.ROUTER)
    sock_sync.bind(sync_addr)    

    id_set = set(ids)
    while True:
        for id in id_set:
            sock.send_multipart([id, ''])
        if sock_sync.poll(timeout):
            id, _ = sock_sync.recv_multipart()

            # Make the last sync message different so that the dealer
            # will know when to stop discarding messages:
            if id in id_set:
                sock.send_multipart([id, 'done'])
                id_set.remove(id)

        if not id_set:
            break

def sync_dealer(sock, id=None, sync_addr='ipc://sync', timeout=10):
    """
    Synchronize a DEALER socket with a single ROUTER socket.
    Must be paired with a call to `sync_router`.

    Parameters
    ----------
    sock : zmq.Socket
        DEALER socket to synchronize.
    id : str
        ID associated with ROUTER synchronization socket. 
        If not specified, the ID is assumed to be the same as that
        of the specified DEALER socket.
    sync_addr : str
        Port address to use for synchronization socket pair. This 
        must be the same as that passed to `sync_router`.
    timeout : int
        Polling timeout.
    """

    assert sock.getsockopt(zmq.TYPE) == zmq.DEALER
    sock_sync = sock.context.socket(zmq.DEALER)
    if id is None:
        sock_sync.setsockopt(zmq.IDENTITY, sock.getsockopt(zmq.IDENTITY))
    else:
        sock_sync.setsockopt(zmq.IDENTITY, id)
    sock_sync.connect(sync_addr)

    while True:
        if sock.poll(timeout):
            
            msg = sock.recv()
            sock_sync.send('')

            # Discard messages until the last synchronization message is
            # received:
            while sock.poll(timeout):
                if sock.recv() == 'done':
                    break
            break
    
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

