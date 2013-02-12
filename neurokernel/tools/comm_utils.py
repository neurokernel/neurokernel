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
    
class ZMQOutput(twiggy.outputs.Output):
    """
    Output messages to a ZeroMQ PUB socket.
    """
    
    def __init__(self, addr, 
                 format=None, close_atexit=True):
        self.addr = addr
        super(ZMQOutput, self).__init__(format, close_atexit)
        
    def _open(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind('tcp://*:5000')

    def _close(self):
        self.sock.close()

    def _write(self, x):
        self.sock.send(x)

