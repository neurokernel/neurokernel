#!/usr/bin/env python
import pycuda.driver as cuda
import Module
import numpy as np
from tools.simpleio import *
import tools.parray as parray
import progressbar as pb
import random as rd

dev1 = 1

cuda.init()

np.random.seed(0)

dt = 1e-4
dur = 1.0
Nt = int(dur / dt)
t = np.arange(0, 1, dt)

# In order to understand pre_neuron, post_neuron and dendrites it's necessary
# notice that the process is over the synapses instead of neurons. So, in fact
# there is no neurons, but connection between neurons.
# Number of dendrites per neuron. A dendrite is a neuron's input connection.
# Shape of num_dendrites: (num_neurons,)
num_dendrites = read_file('example/n_dendrites.h5').astype(np.int32)
num_neurons = num_dendrites.size
# A pre_neuron is the sender neuron's index. Shape: (num_dendrites,)
pre_neuron = read_file('example/pre.h5').astype(np.int32)
num_synapses = pre_neuron.size
# A post_neuron is the receiver neuron's index, and it is organized as a set.
# The elements are organized in crescent order. Shape: (num_dendrites,)
post_neuron = read_file('example/post.h5').astype(np.int32)

# TODO: start_idx is the initial memory address for what? 
start_idx = np.array([0, 768, 1536, 2304, 3072, 3840, 4608, 5376, 6144, 6912,
                   7680, 8448, 9216, 9984, 10752], dtype = np.int32)
num_types = start_idx.size
offset = np.array([ 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0. ], dtype = np.float64)

# Parameters of the model: threshold, slope, saturation, Vs and phy.
# Shape: (num_synapses,)
thres = np.asarray([rd.gauss(-.5, .01) for x in np.zeros([num_synapses])],
                   dtype = np.float64)
slope = np.asarray([rd.gauss(-.5, .1) for x in np.zeros([num_synapses])],
                   dtype = np.float64)
saturation = np.asarray([rd.gauss(.1, .01) for x in np.zeros([num_synapses])],
                        dtype = np.float64)
power = np.ones([num_synapses], dtype = np.float64)
reverse = np.asarray([rd.gauss(-.4, .1) for x in np.zeros([num_synapses])],
                     dtype = np.float64)
V_1 = np.asarray([rd.gauss(.13, .03) for x in np.zeros([num_types])],
                 dtype = np.float64)
V_2 = np.asarray([rd.gauss(.15, .001) for x in np.zeros([num_types])],
                 dtype = np.float64)
V_3 = np.asarray([rd.gauss(-.25, .1) for x in np.zeros([num_types])],
                 dtype = np.float64)
V_4 = np.asarray([rd.gauss(.15, .05) for x in np.zeros([num_types])],
                 dtype = np.float64)
Tphi = np.asarray([rd.gauss(.2, .01) for x in np.zeros([num_types])],
                  dtype = np.float64)

# Parameters of alpha function. Shape: (num_synapses,)
delay = np.ones([num_synapses], dtype = np.float64)

# Initial condition at resting potential. Shape of both: (num_neurons,)
V = np.asarray([rd.gauss(-.51, .01) for x in np.zeros([num_neurons])],
               dtype = np.float64)
n = np.asarray([rd.gauss(.3, .05) for x in np.zeros([num_neurons])],
               dtype = np.float64)

m1 = Module.Module([num_types, num_neurons, 24 * 32, start_idx, num_dendrites,
                    num_synapses, pre_neuron, post_neuron, thres, slope, power,
                    saturation, delay, reverse, dt, V, n, V_1, V_2, V_3, V_4,
                    Tphi, offset, 6], dev1)

#input video
I_ext = parray.to_gpu(np.ones([dur / dt, 4608]))

out = np.empty((Nt, num_neurons), np.double)

playstep = 100
pbar = pb.ProgressBar(maxval = Nt).start()
for i in range(Nt):
    pbar.update(i)
    m1.run_step(in_non_list = int(I_ext.gpudata) + I_ext.dtype.itemsize * \
                I_ext.ld * i, proj_non = out[i, :])

pbar.finish()
