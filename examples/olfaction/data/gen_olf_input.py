#!/usr/bin/env python

"""
Generate sample olfactory model stimulus.
"""

import numpy as np
import h5py

osn_num = 1375

dt = 1e-4 # time step
Ot = 2000 # number of data point during reset period
Rt = 1000 # number of data point during odor delivery period
Nt = 4*Ot + 3*Rt # number of data points in time
t  = np.arange(0, dt*Nt, dt)

I       = 10.0 # amplitude of odorant concentration
u_on    = I*np.ones(Ot, dtype=np.float64)
u_off   = np.zeros(Ot, dtype=np.float64)
u_reset = np.zeros(Rt, dtype=np.float64)
u       = np.concatenate((u_off, u_reset, u_on, u_reset, u_off, u_reset, u_on))
u_all   = np.transpose(np.kron(np.ones((osn_num, 1)), u))

with h5py.File('olfactory_input.h5', 'w') as f:
    f.create_dataset('real', (Nt, osn_num),
                     dtype=np.float64,
                     data=u_all)
