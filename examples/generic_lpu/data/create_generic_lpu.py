#!/usr/bin/env python

"""
Create GEXF file detailing generic lpu.
"""
import numpy as np
import h5py
import random
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt

random.seed(20131224)

# specify numbers of neurons
neu_type  = ('sensory','local','output')
neu_num = map( lambda x: random.randint(30,40), neu_type )

G = nx.DiGraph()

G.add_nodes_from(range(sum(neu_num)))

idx = 0
for (t,n) in zip(neu_type,neu_num):
    for i in range(n):
        name = t+"_"+str(i)
        if t != 'local' or random.random() < 0.5:
            G.node[idx] = {
                'model':'LeakyIAF',
                'name':name+'_s',
                'input':True if t == 'sensory' else False,
                'public':True if t == 'output' else False,
                'spiking':True,
                'V':random.uniform(-0.06,-0.025),
                'Vr':-0.0675489770451,
                'Vt':-0.0251355161007,
                'R':1.02445570216,
                'C':0.0669810502993}
        else:
            G.node[idx] = {
                'model':"MorrisLecar",
                'name':name+'_g',
                'input':True if t == 'sensory' else False,
                'public':True if t == 'output' else False,
                'spiking':False,
                'V1':0.3,
                'V2':0.15,
                'V3':0,
                'V4':0.3,
                'phi':0.025,
                'offset':0,
                'initV':-0.5214,
                'initn':0.03,
                'n_dendrites':1}
        idx += 1

for r,(i,j) in zip((0.5,0.1,0.1),((0,1),(0,2),(1,2))):
    src_off = sum( neu_num[0:i] )
    tar_off = sum( neu_num[0:j] )
    for src,tar in product( range( src_off, src_off+neu_num[i]),
                            range( tar_off, tar_off+neu_num[j]) ):
        if random.random() > r: continue
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

nx.write_gexf(G,"generic_lpu.gexf.gz")

# create input stimulus for sensory neurons
dt = 1e-4 # time step
dur = 1.0
Nt = int(dur/dt)
t  = np.arange(0, dt*Nt, dt)

I  = np.zeros((Nt,neu_num[0]),dtype=np.float64)
I[ np.logical_and( t>0.3,t<0.6) ] = 0.6

with h5py.File('generic_input.h5', 'w') as f:
    f.create_dataset('array', (Nt, neu_num[0]),
                     dtype=np.float64,
                     data=I)
