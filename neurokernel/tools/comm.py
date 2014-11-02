#!/usr/bin/env python

"""
Communication utilities.
"""

from mpi4py import MPI
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

class MPIOutput(twiggy.outputs.Output):
    """
    Output messages to a file via MPI I/O.
    """
    def __init__(self, name, format, comm,
                 mode=MPI.MODE_CREATE | MPI.MODE_WRONLY,
                 close_atexit=True):
        self.filename = name
        self._format = format if format is not None else self._noop_format
        self.comm = comm
        self.mode = mode
        super(MPIOutput, self).__init__(format, close_atexit)

    def _open(self):
        self.file = MPI.File.Open(self.comm, self.filename,
                                  self.mode)

    def _close(self):
        self.file.Close()

    def _write(self, x):
        self.file.Iwrite_shared(x)
    
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

