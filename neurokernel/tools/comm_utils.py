import zmq

def is_poll_in(sock, poller, timeout=100):
    """
    Check for incoming data on a socket using a poller.
    """

    socks = dict(poller.poll(timeout))
    if s in socks and socks[s] == zmq.POLLIN:
        return True
    else:
        return False
    
