#!/usr/bin/env python

"""
Logging tools.
"""

import sys
import traceback

import twiggy

from tools.comm import ZMQOutput

def format_name(name, width=20):
    """
    Pad process name with spaces.

    Parameters
    ----------
    name : str
        Name to pad.
    width : int
        Total width of padded name.

    Returns
    -------
    padded : str
        Padded name.
    """

    return ('{name:%s}' % width).format(name=name)

def log_exception(type, value, tb):
    """
    Log the specified exception data using twiggy.
    """

    msg = '|'.join([': '.join([y.strip() for y in x.strip('\n ').split('\n')]) for x in \
                    traceback.format_exception(type, value, tb)[1:]])
    twiggy.log.error('Uncaught exception: %s' % str(msg))

def setup_logger(name='', level=twiggy.levels.DEBUG,
                 fmt=twiggy.formats.line_format,
                 stdout=None, file_name=None, sock=None,
                 log_exceptions=True):
    """
    Setup a twiggy logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : twiggy.levels.LogLevel
        Logging level.
    fmt : str
        Format string.
    stdout : bool
        Create output stream handler to stdout if True.
    file_name : str
        Create output handler to specified file.
    sock : str
        ZeroMQ socket address.
    log_exceptions : bool
        If True, exception messages are written to the logger.

    Returns
    -------
    logger : twiggy.logger.Logger
        Configured logger.
    """

    if file_name:
        file_output = \
          twiggy.outputs.FileOutput(file_name, fmt, 'w')
        twiggy.addEmitters(('file', level, None, file_output))

    if stdout:
        stdout_output = \
          twiggy.outputs.StreamOutput(fmt, stream=stdout)   
        twiggy.addEmitters(('stdout', level, None, stdout_output))

    if sock:
        port_output = ZMQOutput(sock, fmt)
        twiggy.addEmitters(('sock', level, None, sock_output))

    if log_exceptions:
        sys.excepthook = log_exception

    return twiggy.log.name(format_name(name))
