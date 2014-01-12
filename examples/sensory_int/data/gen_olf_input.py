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
#Nt = 4*Ot + 3*Rt # number of data points in time

#Nt = 10000
#t  = np.arange(0, dt*Nt, dt)

I       = 0.5195 # amplitude of odorant concentration
u_1 = np.zeros(500, np.float64)
u_2 = I*np.ones(5000, np.float64)
u_3 = np.zeros(4500,np.float64)
u_4 = I*np.ones(1000,np.float64)
u_5 = np.zeros(1000, np.float64)
u_6 = I*np.ones(1500, np.float64)
u_7 = np.zeros(500,np.float64)
#u_on    = I*np.ones(Ot, dtype=np.float64)
#u_off   = np.zeros(Ot, dtype=np.float64)
#u_reset = np.zeros(Rt, dtype=np.float64)
u       = np.concatenate((u_1,u_2,u_3,u_4,u_5,u_6,u_7))
Nt = u.size
#print Nt
u_all   = np.transpose(np.kron(np.ones((osn_num, 1)), u))

with h5py.File('olfactory_input.h5', 'w') as f:
    f.create_dataset('array', (Nt, osn_num),
                     dtype=np.float64,
                     data=u_all)
