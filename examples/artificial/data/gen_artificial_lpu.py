#!/usr/bin/env python

"""
Create artificial LPU and simple pulse input signal.
"""

from itertools import product
import random
import sys

import numpy as np
import h5py
import networkx as nx

def create_lpu(file_name, N_sensory, N_local, N_output):

    # Set numbers of neurons:
    neu_type = ('sensory', 'local', 'output')
    neu_num = (N_sensory, N_local, N_output)

    G = nx.DiGraph()

    G.add_nodes_from(range(sum(neu_num)))

    idx = 0
    for (t, n) in zip(neu_type, neu_num):
        for i in range(n):
            name = t+"_"+str(i)
            if t != 'local' or np.random.rand() < 0.5:
                G.node[idx] = {
                    'model': 'LeakyIAF',
                    'name': name+'_s',
                    'input': True if t == 'sensory' else False,
                    'public': True if t == 'output' else False,
                    'spiking': True,
                    'V': np.random.uniform(-0.06,-0.025),
                    'Vr': -0.0675489770451,
                    'Vt': -0.0251355161007,
                    'R': 1.02445570216,
                    'C': 0.0669810502993}
            else:
                G.node[idx] = {
                    'model': "MorrisLecar",
                    'name': name+'_g',
                    'input': True if t == 'sensory' else False,
                    'public': True if t == 'output' else False,
                    'spiking': False,
                    'V1': 0.3,
                    'V2': 0.15,
                    'V3': 0,
                    'V4': 0.3,
                    'phi': 0.025,
                    'offset': 0,
                    'initV': -0.5214,
                    'initn': 0.03,
                    'n_dendrites': 1}
            idx += 1

    for r, (i, j) in zip((0.5, 0.1, 0.1, 0.3),
                         ((0, 1), (0, 2), (1, 2), (2,1))):
        src_off = sum(neu_num[0:i])
        tar_off = sum(neu_num[0:j])
        for src,tar in product( range( src_off, src_off+neu_num[i]),
                                range( tar_off, tar_off+neu_num[j]) ):
            if np.random.rand() > r: continue
            name = G.node[src]['name'] + '-' + G.node[tar]['name']
            if G.node[src]['name'][-1] == 's':
                G.add_edge(src,tar,type='directed',attr_dict={
                    'model'       : 'AlphaSynapse',
                    'name'        : name,
                    'class'       : 0 if G.node[tar]['name'][-1] == 's' else 1,
                    'ar'          : 1.1*1e2,
                    'ad'          : 1.9*1e3,
                    'reverse'     : 65*1e-3,
                    'gmax'        : 3*1e-3,
                    'conductance' : True})
            else:
                G.add_edge(src,tar,type='directed',attr_dict={
                    'model'       : 'power_gpot_gpot',
                    'name'        : name,
                    'class'       : 2 if G.node[tar]['name'][-1] == 's' else 3,
                    'slope'       : 4e9,
                    'threshold'   : -0.06,
                    'power'       : 4,
                    'saturation'  : 30,
                    'delay'       : 1,
                    'reverse'     : -0.015,
                    'conductance' : True})

    nx.write_gexf(G, file_name)

def create_input(file_name, N_sensory, dt=1e-4, dur=1.0, start=0.3, stop=0.6, I_max=0.6):
    """
    Create input stimulus for sensory neurons in artificial LPU.

    Creates an HDF5 file containing input signals for the specified number of
    neurons. The signals consist of a rectangular pulse of specified duration
    and magnitude.

    Parameters
    ----------
    file_name : str
        Name of output HDF5 file.
    N_sensory : int
        Number of sensory neurons.
    dt : float
        Time resolution of generated signal.
    dur : float
        Duration of generated signal.
    start : float
        Start time of signal pulse.
    stop : float
        Stop time of signal pulse.
    I_max : float
        Pulse magnitude.
    """

    Nt = int(dur/dt)
    t  = np.arange(0, dt*Nt, dt)

    I  = np.zeros((Nt, N_sensory), dtype=np.float64)
    I[np.logical_and(t>start, t<stop)] = I_max

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('array', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='artificial_lpu.gexf.gz',
                        help='LPU file name')
    parser.add_argument('in_file_name', nargs='?', default='artificial_input.h5',
                        help='Input file name')
    parser.add_argument('-s', type=int,
                        help='Seed random number generator')
    args = parser.parse_args()

    if args.s is not None:
        np.random.seed(args.s)
    dt = 1e-4
    dur = 1.0
    start = 0.3
    stop = 0.6
    I_max = 0.6
    neu_num = [np.random.randint(30, 41) for i in xrange(3)]

    create_input(args.in_file_name, neu_num[0], dt, dur, start, stop, I_max)
    create_lpu(args.lpu_file_name, *neu_num)
