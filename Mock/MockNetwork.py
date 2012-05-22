import pycuda.gpuarray as garray
import pycuda.driver as cuda
import tools.parray as parray
import numpy as np
import random as rd
from tools.simpleio import *
from MorrisLecar import MorrisLecar
from VectorSynapse import VectorSynapse
from ..Module as Mod

class MockNetwork (Mod):
    """
    Neural network class. This code, by now, is provided by the user. In this
    example, this code is the lamina version implemented by Nikul and Yiyin.
    """
    def __init__(self, manager, dt, num_in_non, num_in_spike, num_proj_non,
                 num_proj_spike, device):

        np.random.seed(0)

        Mod.__init__(self, manager, dt, num_in_non, num_in_spike, num_proj_non,
                 num_proj_spike, device)

        # In order to understand pre_neuron, post_neuron and dendrites it's
        # necessary notice that the process is over the synapses instead of
        # neurons. So, in fact there is no neurons, but connection between
        # neurons. Number of dendrites per neuron. A dendrite is a neuron's
        # input connection. Shape of num_dendrites: (num_neurons,)
        num_dendrites = read_file('example/n_dendrites.h5').astype(np.int32)
        num_neurons = num_dendrites.size
        # A pre_neuron is the sender neuron's index. Shape: (num_dendrites,)
        pre_neuron = read_file('example/pre.h5').astype(np.int32)
        num_synapses = pre_neuron.size
        # A post_neuron is the receiver neuron's index, and it is organized as
        # a set. The elements are organized in crescent order. Shape:
        # (num_dendrites,)
        post_neuron = read_file('example/post.h5').astype(np.int32)

        # TODO: start_idx is the initial memory address for what? 
        start_idx = np.array([0, 768, 1536, 2304, 3072, 3840, 4608, 5376, 6144,
                              6912, 7680, 8448, 9216, 9984, 10752],
                             dtype = np.int32)
        num_types = start_idx.size
        offset = np.array([ 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0.2, 0.2, 0.2,
                           0.2, 0.2, 0.2, 0.2, 0. ], dtype = np.float64)

        # Parameters of the model: threshold, slope, saturation, Vs and phy.
        # Shape: (num_synapses,)
        thres = np.asarray([rd.gauss(-.5, .01) for x in \
                            np.zeros([num_synapses])], dtype = np.float64)
        slope = np.asarray([rd.gauss(-.5, .1) for x in \
                            np.zeros([num_synapses])], dtype = np.float64)
        saturation = np.asarray([rd.gauss(.1, .01) for x in \
                                 np.zeros([num_synapses])], dtype = np.float64)
        power = np.ones([num_synapses], dtype = np.float64)
        reverse = np.asarray([rd.gauss(-.4, .1) for x in \
                              np.zeros([num_synapses])], dtype = np.float64)
        V_1 = np.asarray([rd.gauss(.13, .03) for x in \
                          np.zeros([num_types])], dtype = np.float64)
        V_2 = np.asarray([rd.gauss(.15, .001) for x in \
                          np.zeros([num_types])], dtype = np.float64)
        V_3 = np.asarray([rd.gauss(-.25, .1) for x in \
                          np.zeros([num_types])], dtype = np.float64)
        V_4 = np.asarray([rd.gauss(.15, .05) for x in \
                          np.zeros([num_types])], dtype = np.float64)
        Tphi = np.asarray([rd.gauss(.2, .01) for x in \
                           np.zeros([num_types])], dtype = np.float64)

        # Parameters of alpha function. Shape: (num_synapses,)
        delay = np.ones([num_synapses], dtype = np.float64)

        # Initial condition at resting potential. Shape of both: (num_neurons,)
        V = np.asarray([rd.gauss(-.51, .01) for x in np.zeros([num_neurons])],
                       dtype = np.float64)
        n = np.asarray([rd.gauss(.3, .05) for x in np.zeros([num_neurons])],
                       dtype = np.float64)

        self.num_neurons = num_neurons
        self.dt = dt

        delay_steps = int(round(max(delay) * 1e-3 / dt))

        self.buffer = CircularArray(num_neurons, delay_steps, V)

        self.neurons = MorrisLecar(num_neurons, num_types, 24 * 32,
                                   start_idx, dt, num_dendrites, V, n, V_1,
                                   V_2, V_3, V_4, Tphi, offset, 6)
        self.synapses = VectorSynapse(num_synapses, pre_neuron, post_neuron,
                                      thres, slope, power, saturation, delay,
                                      reverse, dt)

    def __run_step(self, in_non_list = None, in_spike_list = None,
                 proj_non = None, proj_spike = None):

        self.neurons.I_pre.fill(0)
        self.neurons.update_I_pre_input(in_non_list)

        self.neurons.read_synapse(self.synapses.conductance,
                                  self.synapses.V_rev)

        self.neurons.eval(self.buffer)

        self.synapses.compute_synapse(self.buffer)

        cuda.memcpy_dtoh(proj_non, self.neurons.V.gpudata)
        self.buffer.step()

class CircularArray:
    def __init__(self, num_neurons, delay_steps, rest):
        self.dtype = np.double
        self.num_neurons = num_neurons
        self.delay_steps = delay_steps

        self.buffer = parray.empty((delay_steps, num_neurons), np.double)

        d_rest = garray.to_gpu(rest)
        self.current = 0

        #initializing V buffer
        for i in range(delay_steps):
            cuda.memcpy_dtod(int(self.buffer.gpudata) + self.buffer.ld * i,
                             d_rest.gpudata, d_rest.nbytes)

    def step(self):
        self.current += 1
        if self.current >= self.delay_steps:
            self.current = 0

