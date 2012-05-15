import pycuda.driver as cuda
import Network as nn
import atexit
import numpy as np
import pycuda.gpuarray as garray
from simpleio import *
import parray
import time
import getpass
import progressbar as pb

dev = 1

cuda.init()
ctx = cuda.Device(dev).make_context()
atexit.register(ctx.pop)

np.random.seed(0)

dt = 1e-4
dur = 1.0
Nt = int(dur / dt)
t = np.arange(0, 1, dt)

# Number of dendrites per neuron
num_dendrites = read_file('example/n_dendrites.h5').astype(np.int32)
num_neurons = num_dendrites.size
pre_neuron = read_file('example/pre.h5').astype(np.int32)
num_synapse = pre_neuron.size
post_neuron = read_file('example/post.h5').astype(np.int32)
thres = read_file('example/threshold.h5')
slope = read_file('example/slope.h5')
power = read_file('example/power.h5')
saturation = read_file('example/saturation.h5')
delay = read_file('example/delay.h5')
reverse = read_file('example/reverse.h5')

start_idx = read_file('example/start_idx.h5').astype(np.int32)
num_types = start_idx.size
V_1 = read_file('example/V1.h5')
V_2 = read_file('example/V2.h5')
V_3 = read_file('example/V3.h5')
V_4 = read_file('example/V4.h5')
Tphi = read_file('example/T.h5')
offset = read_file('example/offset.h5')

#intial condition at resting potential
V = read_file('example/initV.h5')
n = read_file('example/initn.h5')

network = nn.Network(num_types, num_neurons, 24 * 32, start_idx, num_dendrites,
                     num_synapse, pre_neuron, post_neuron, thres, slope, power,
                     saturation, delay, reverse, dt, V, n, V_1, V_2, V_3, V_4,
                     Tphi, offset, 6)

#input video
h_I_ext = np.zeros([10000, 4608])

I_ext = parray.to_gpu(h_I_ext)

out = np.empty((Nt, num_neurons), np.double)

#turn this on to allow visual output
prin = False

playstep = 100
for i in range(Nt):
    network.run_step(int(I_ext.gpudata) + I_ext.dtype.itemsize * I_ext.ld * i,
                     out[i, :])
