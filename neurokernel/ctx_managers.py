#!/usr/bin/env python

"""
Various signal-related context managers
"""

from contextlib import contextmanager
import signal

@contextmanager
def ExceptionOnSignal(s=signal.SIGUSR1, e=Exception, i=None):
    """
    Raise a specific exception when the specified signal is detected.
    """

    def handler(signum, frame):
        if i is not None:
            raise e('signal %i detected in %s' % (s, i))
        else:
            raise e('signal %i detected' % s)
    signal.signal(s, handler)
    yield

@contextmanager
def TryExceptionOnSignal(s=signal.SIGUSR1, e=Exception, i=None):
    """
    Check for exception raised in response to specific signal.
    """

    with ExceptionOnSignal(s, e, i):
        try:
            yield
        except e:
            pass

@contextmanager
def IgnoreSignal(s=signal.SIGUSR1):
    """
    Ignore the specified signal.
    """

    signal.signal(s, signal.SIG_IGN)
    yield

@contextmanager
def IgnoreKeyboardInterrupt():
    """
    Ignore keyboard interrupts.
    """

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    yield

@contextmanager
def OnKeyboardInterrupt(handler):
    """
    Respond to keyboard interrupt with specified handler.
    """

    signal.signal(signal.SIGINT, handler)
    yield

if __name__ == '__main__':

    # This example should quit when Ctrl-C is pressed:
    import time
    def handler(signum, frame):
        print 'caught'
        handler.done = True
    handler.done = False

    with OnKeyboardInterrupt(handler):
        while True:
            print 'waiting'
            time.sleep(1)
            if handler.done:
                break
