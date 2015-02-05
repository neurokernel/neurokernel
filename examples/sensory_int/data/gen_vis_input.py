#!/usr/bin/env python

"""
Generate sample vision model stimulus.
"""

import numpy as np
import h5py

n_col = 32
n_row = 24

dt = 1e-4
a = np.zeros(14000, np.float64)
a[2000:8000] = 0.016
a[8500:13500] = 0.016

S = np.zeros((n_row, n_col, 14000), np.float64)
x = np.arange(n_col)*1.5
y = np.arange(n_row)*np.sqrt(3)

div = 0
signv = 1
dih = 0
signh = 1

for i in xrange(14000):
    div = div -0.1
    if div < 0:
        div = x[-1]
        signv = 1 - signv
    dih = dih - 0.1
    if dih < 0:
        dih = y[-1]
        signh = 1 - signh

    if a[i] != 0:
        if i<= 8000:
            if signv:
                c = 1 - np.asarray(x > div, np.double)
            else:
                c = np.asarray(x > div, np.double)
            I = np.tile(c, [n_row, 1])
        else:
            if signh:
                c = np.reshape(1 - np.asarray(y > dih, np.double), [n_row, 1])
            else:
                c = np.reshape(np.asarray(y > dih, np.double), [n_row, 1])
            I = np.tile(c, [1, n_col])
        b = I*a[i]
        S[:, :, i] = b

A = S.reshape((n_row*n_col, 14000))
A = np.tile(A, [6, 1]).T
with h5py.File('vision_input.h5', 'w') as f:
    f.create_dataset('array', A.shape, dtype=A.dtype, data=A)
