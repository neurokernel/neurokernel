#!/usr/bin/env python

import numpy as np
import sys, traceback

def rand_bin_matrix(sh, N, dtype=np.double):
    """
    Generate a rectangular binary matrix with randomly distributed nonzero entries.

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

        disp(func.__name__ + ': ' + e.message + \
           ' (' + fname + ':' + str(lineno) + ')')
             
