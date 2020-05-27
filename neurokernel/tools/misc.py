#!/usr/bin/env python

from functools import wraps
import itertools
import numbers
import re
import subprocess
import sys
import traceback

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
import numpy as np

try:
    from subprocess import DEVNULL
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

def get_open_files(*pids):
    """
    Find files opened by specified process ID(s).

    Parameters
    ----------
    pids : list of int
        Process IDs.

    Returns
    -------
    files : list of str
        Open file names.
    """

    for pid in pids:
        if not isinstance(pid, numbers.Integral):
            raise ValueError('invalid PID')
    files = set()
    for pid in pids:
        try:
            out = subprocess.check_output(['lsof', '-wXFn', '+p', str(pid)],
                    stderr=DEVNULL)
        except:
            pass
        else:
            lines = out.strip().split('\n')
            for line in lines:

                # Skip sockets, pipes, etc.:
                if line.startswith('n') and line[1] == '/':
                    files.add(line[1:])
    return list(files)

def get_pids_open(*files):
    """
    Find processes with open handles for the specified file(s).

    Parameters
    ----------
    files : list of str
        File paths.

    Returns
    -------
    pids : list of int
        Process IDs with open handles to the specified files.
    """

    for f in files:
        if not isinstance(f, basestring):
            raise ValueError('invalid file name %s' % f)
    pids = set()
    try:
        out = subprocess.check_output(['lsof', '+wt']+list(files),
                stderr=DEVNULL)
    except Exception as e:
        out = str(e.output)
    if not out.strip():
        return []
    lines = out.strip().split('\n')
    for line in lines:
        pids.add(int(line))
    return list(pids)

def rand_bin_matrix(sh, N, dtype=np.double):
    """
    Generate a rectangular binary matrix with randomly distributed nonzero entries.

    Examples
    --------
    >>> m = rand_bin_matrix((2, 3), 3)
    >>> set(m.flatten()) == set([0, 1])
    True

    Parameters
    ----------
    sh : tuple
        Shape of generated matrix.
    N : int
        Number of entries to set to 1.
    dtype : dtype
        Generated matrix data type.

    """

    result = np.zeros(sh, dtype)
    indices = np.arange(result.size)
    np.random.shuffle(indices)
    result.ravel()[indices[:N]] = 1
    return result

def catch_exception(func, disp, debug, *args, **kwargs):
    """
    Catch and report exceptions when executing a function.

    If an exception occurs while executing the specified function, the
    exception's message and the line number where it occurred (in the innermost
    traceback frame) are displayed.

    Examples
    --------
    >>> import sys
    >>> def f(x): x/0
    >>> catch_exception(f, sys.stdout.write, 1) # doctest: +ELLIPSIS
    f: integer division or modulo by zero (...:1)

    Parameters
    ----------
    func : function
        Function to execute.
    disp : function
        Function to use to display exception message.
    args : list
        Function arguments.
    kwargs : dict
        Named function arguments.

    """

    try:
        func(*args, **kwargs)
    except Exception as e:

        # Find the line number of the innermost traceback frame:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        # error = traceback.format_exception(exc_type, exc_value,
        #                                    exc_traceback)
        for fname in traceback.extract_tb(exc_traceback):
            fname, lineno, fn, text = fname
        disp(func.__name__ + ': ' + e.__class__.__name__ + ': ' + str(e) + \
           ' (' + fname + ':' + str(lineno) + ')')
        if debug:
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      file=sys.stderr)
            raise e
        return traceback.format_exception(exc_type, exc_value, exc_traceback)

def memoized_property(fget):
    """
    Decorator for creating a property that only calls its getter once.

    Notes
    -----
    Copied from https://github.com/estebistec/python-memoized-property
    under the BSD license.
    """

    attr_name = '_{0}'.format(fget.__name__)
    @wraps(fget)
    def fget_memoized(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fget(self))
        return getattr(self, attr_name)
    return property(fget_memoized)

def dtype_to_mpi(t):
    """
    Convert Numpy data type to MPI type.

    Parameters
    ----------
    t : type
        Numpy data type.

    Returns
    -------
    m : mpi4py.MPI.Datatype
        MPI data type corresponding to `t`.
    """

    if hasattr(MPI, '_typedict'):
        m = MPI._typedict[np.dtype(t).char]
    elif hasattr(MPI, '__TypeDict__'):
        m = MPI.__TypeDict__[np.dtype(t).char]
    else:
        raise ValueError('cannot convert type')
    return m

def openmpi_cuda_support(path='ompi_info'):
    """
    Check whether CUDA support is available in OpenMPI.

    Parameters
    ----------
    path : str
        Path to ompi_info binary.

    Returns
    -------
    result : bool
        True if OpenMPI was built with CUDA support.
    """

    try:
        out = subprocess.check_output([path, '-l', '9', '--param', 'mpi', 'all',
                                       '--parsable'])
    except:
        return False
    else:
        lines = out.split('\n')
        for line in lines:
            if re.search(r'mpi_built_with_cuda_support\:value', line):
                tokens = line.split(':')
                if tokens[-1] == 'true':
                    return True
                else:
                    return False
        return False

def renumber_in_order(arr):
    """
    Map unique array elements to in-order integers.

    Maps an array of elements that may contain duplicates and might not be
    monotonically ordered to an array of equivalent length where the unique elements
    are respectively replaced by 0, 1, etc.

    Parameters
    ----------
    arr : array_like
        1D array of elements.

    Returns
    -------
    result : list
        Array of mapped integers.

    Examples
    --------
    >>> arr = np.array([0, 2, 2, 3, 5, 5])
    >>> result = renumber_in_order(arr)
    >>> np.allclose(arr, [0, 1, 1, 2, 3, 3])
    True
    """

    c = itertools.count()
    result = []
    already_seen = {}
    for e in arr:
        if e not in already_seen:
            already_seen[e] = next(c)
        result.append(already_seen[e])
    return result


class LPUExecutionError(Exception):
    pass
