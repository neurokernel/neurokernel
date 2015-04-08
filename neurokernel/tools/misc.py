#!/usr/bin/env python

from functools import wraps
import re
import subprocess
import sys
import traceback

from mpi4py import MPI
import numpy as np

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

def catch_exception(func, disp, *args, **kwargs):
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
        for fname in traceback.extract_tb(sys.exc_info()[2]):
            fname, lineno, fn, text = fname

        disp(func.__name__ + ': ' + e.__class__.__name__ + ': ' + str(e.message) + \
           ' (' + fname + ':' + str(lineno) + ')')
             
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
    
