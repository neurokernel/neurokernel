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

def log_exception(type, value, tb, logger=twiggy.log, multiline=False):
    """
    Log the specified exception data using twiggy.

    Parameters
    ----------
    value, tb : object
        Parameters expected by traceback.print_exception.
    logger : twiggy.logger.Logger
        Logger to use. twiggy.log is assumed by default.
    multiline : bool
        If True, print exception using multiple log lines.
    """

    if multiline:
        for x in traceback.format_exception(type, value, tb)[1:]:
            for y in x.strip('\n ').split('\n'):
                logger.error(y)
    else:
        msg = '|'.join([': '.join([y.strip() for y in x.strip('\n ').split('\n')]) for x in \
                        traceback.format_exception(type, value, tb)[1:]])
        logger.error('Uncaught exception: %s' % str(msg))

def set_excepthook(logger, multiline=False):
    """
    Set the exception hook to use the specified logger.

    Parameters
    ----------
    logger : twiggy.logger.Logger
        Configured logger.
    multiline : bool
        If True, log exception messages on multiple lines.

    """

    sys.excepthook = \
        lambda type, value, tb: log_exception(type, value, tb, logger, multiline)

def setup_logger(name='', level=twiggy.levels.DEBUG,
                 fmt=twiggy.formats.line_format,
                 stdout=None, file_name=None, sock=None,
                 log_exceptions=True, multiline=False):
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
    multiline : bool
        If True, log exception messages on multiple lines.

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

    logger = twiggy.log.name(format_name(name))
    if log_exceptions:
        set_excepthook(logger, multiline)

    return logger
