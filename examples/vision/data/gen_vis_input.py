#!/usr/bin/env python

"""
Generate sample vision model stimulus.
"""

import numpy as np
import h5py

n_col = 32
n_row = 24

dt = 1e-4
a = np.zeros(11000, np.float64)
a[0:2000] = 0.16
a[6000:8000] = 0.16

S = np.zeros((n_row, n_col, 11000), np.float64)
for i in xrange(11000):
    if a[i] != 0:
        c = np.sign(np.sin(8*np.arange(32)*np.pi/180+16*np.pi*i*dt))
        I = np.tile(c, [n_row, 1])
        b = I*a[i]
        S[:, :, i] = b

A = S.reshape((n_row*n_col, 11000), order='F')
A = np.tile(A, [6, 1]).T
with h5py.File('vision_input.h5', 'w') as f:
    f.create_dataset('real', A.shape, dtype=A.dtype, data=A)
