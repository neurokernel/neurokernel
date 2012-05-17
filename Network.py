import pycuda.gpuarray as garray
import pycuda.driver as cuda
import numpy as np
import time
import tools.parray as parray
from MorrisLecar import MorrisLecar
from VectorSynapse import VectorSynapse

class Network:
    """
    Neural network class.
    """
    def __init__(self, in_non_list, in_spike_list, proj_non, proj_spike, param):
        num_types = param[0]
        num_neurons = param[1]
        num_cart = param[2]
        neuron_start = param[3]
        num_dendrite = param[4]
        num_synapse = param[5]
        pre_neuron = param[6]
        post_neuron = param[7]
        syn_thres = param[8]
        syn_slope = param[9]
        syn_power = param[10]
        syn_saturation = param[11]
        syn_delay = param[12]
        V_rev = param[13]
        dt = param[14]
        V = param[15]
        n = param[16]
        V_1 = param[17]
        V_2 = param[18]
        V_3 = param[19]
        V_4 = param[20]
        Tphi = param[21]
        offset = param[22]
        non_input_start = param[23]

        self.num_neurons = num_neurons
        self.dt = dt

        delay_steps = int(round(max(syn_delay) * 1e-3 / dt))

        self.buffer = CircularArray(num_neurons, delay_steps, V)

        self.neurons = MorrisLecar(num_neurons, num_types, num_cart,
                                   neuron_start, dt, num_dendrite, V, n, V_1,
                                   V_2, V_3, V_4, Tphi, offset, non_input_start)
        self.synapses = VectorSynapse(num_synapse, pre_neuron, post_neuron,
                                      syn_thres, syn_slope, syn_power,
                                      syn_saturation, syn_delay, V_rev, dt)

        self.st1 = None
        self.st2 = None

    def run_step(self, I_ext, out, put = False):

        self.neurons.I_pre.fill(0)
        self.neurons.update_I_pre_input(I_ext)

        self.neurons.read_synapse(self.synapses.conductance,
                                  self.synapses.V_rev)

        self.neurons.eval(self.buffer)

        self.synapses.compute_synapse(self.buffer)

        cuda.memcpy_dtoh(out, self.neurons.V.gpudata)
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
