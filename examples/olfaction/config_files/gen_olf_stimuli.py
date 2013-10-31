"""
Create odorant stimuli in hd5 format
"""

"""
Create the gexf configuration based on E. Hallem's cell paper on 2006

"""

import numpy as np
import h5py

osn_num = 1375;
f = h5py.File("al.hdf5","w")

dt = 1e-4 # time step
Ot = 2000 # number of data point during reset period
Rt = 1000 # number of data point during odor delivery period
Nt = 4*Ot + 3*Rt # number of data point
t  = np.arange(0,dt*Nt,dt)

I       = -1.*0.0195 # amplitude of the onset odorant concentration
u_on    = I*np.ones( Ot, dtype=np.float64)
u_off   = np.zeros( Ot, dtype=np.float64)
u_reset = np.zeros( Rt, dtype=np.float64)
u       = np.concatenate((u_off,u_reset,u_on,u_reset,u_off,u_reset,u_on))
u_all   = np.transpose( np.kron( np.ones((osn_num,1)), u))

# create the dataset
dset = f.create_dataset("acetone_on_off.hdf5",(Nt, osn_num), dtype=np.float64,\
        data = u_all)

f.close()
