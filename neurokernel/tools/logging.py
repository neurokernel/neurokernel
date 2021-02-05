#!/usr/bin/env python

"""
Logging tools.
"""

import copy
import sys
import traceback

import twiggy

# Try to import various dependencies here so that they can be serialized
# and transmitted to mpi_backend during model execution:
try:
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    import mpi4py.MPI
except ImportError:
    pass
try:
    import neurokernel.tools.zmq
except ImportError:
    pass
try:
    import neurokernel.tools.mpi
except ImportError:
    pass

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
                 fmt_name=('{0:%s}' % 10).format,
                 screen=None, file_name=None,
                 mpi_comm=None, zmq_addr=None,
                 log_exceptions=True, multiline=False):
    """
    Setup a twiggy logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : twiggy.levels.LogLevel
        Logging level.
    fmt : twiggy.formats.LineFormat
        Logging formatter class instance.
    fmt_name : function
        Function with one parameter that formats the message name.
    screen : bool
        Create output stream handler to the screen if True.
    file_name : str
        Create output handler to specified file.
    mpi_comm : mpi4py.MPI.Intracomm
        If not None, use MPI I/O with the specified communicator for
        output file handler. Ignored if the `file_name` parameter
        is not specified.
    zmq_addr : str
        ZeroMQ socket address.
    log_exceptions : bool
        If True, exception messages are written to the logger.
    multiline : bool
        If True, log exception messages on multiple lines.

    Returns
    -------
    logger : twiggy.logger.Logger
        Configured logger.

    Bug
    ---
    To use the ZeroMQ output class with multiprocessing, it must be added
    as an emitter within each process.
    """

    fmt = copy.copy(fmt)
    fmt.conversion.delete('name')

    # Apply name format to the value (i.e., the name), not the key (i.e., the
    # field name "name"):
    fmt.conversion.add('name', str, lambda k, v: fmt_name(v))

    if file_name:
        if mpi_comm:
            if 'mpi4py.MPI' not in sys.modules:
                raise ValueError('mpi4py not available')
            if not isinstance(mpi_comm, mpi4py.MPI.Intracomm):
                raise ValueError('mpi_comm must be an instance of '
                                 'mpi4py.MPI.Intracomm')
            if 'neurokernel.tools.mpi' not in sys.modules:
                raise ValueError('neurokernel.tools.mpi not available')
            file_output = \
                neurokernel.tools.mpi.MPIOutput(file_name, fmt, mpi_comm)
        else:
            file_output = \
                twiggy.outputs.FileOutput(file_name, fmt, 'w')
        twiggy.add_emitters(('file', level, None, file_output))

    if screen:
        screen_output = \
          twiggy.outputs.StreamOutput(fmt, stream=sys.stdout)
        twiggy.add_emitters(('screen', level, None, screen_output))

    if zmq_addr:
        if 'neurokernel.tools.zmq' not in sys.modules:
            raise ValueError('neurokernel.tools.zmq not available')
        zmq_output = neurokernel.tools.zmq.ZMQOutput(zmq_addr, fmt)
        twiggy.add_emitters(('zmq', level, None, zmq_output))

    logger = twiggy.log.name(fmt_name(name))
    if log_exceptions:
        set_excepthook(logger, multiline)

    return logger
