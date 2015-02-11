#!/usr/bin/env python

"""
Simple script for checking installed CUDA version.
"""

import ctypes

try:
    _libcudart = ctypes.cdll.LoadLibrary('libcudart.so')
except:
    print 'CUDA runtime library not found'
else:
    _libcudart.cudaDriverGetVersion.restype = int
    _libcudart.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version = ctypes.c_int()
    status = _libcudart.cudaDriverGetVersion(ctypes.byref(version))
    if status != 0:
        print 'CUDA runtime library found: version unknown'
    else:
        major = version.value/1000
        minor = (version.value%1000)/10
        print 'CUDA runtime library found: version %s' % (major + minor/10.0)
