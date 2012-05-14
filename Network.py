#!/usr/bin/env python
import numpy as np
from multiprocessing import Process
import time
import parray
from MorrisLecar import MorrisLecar
from VectorSynapse import VectorSynapse

"""
Can you please construct an object-oriented PyCUDA implementation of a network of ML neurons randomly connected by alpha function synapses (similar to the IAF network you implemented for E9070) that we can use for testing the architecture?
I imagine that a standard numerical ODE solver such as low-order Runge-Kutta should be sufficient for simulating the network.
Also, I'm not sure what dynamics we should expect to observe for such a network. You may need to talk to Yiyin or Nikul for further information about both of these points.
"""

class Network (Process):
    """
    Neural network class.
    
    Parameters
    ----------
    dt : float
        Time resolution of simulation (in ms)
    N : int
        Number of entries in simulation; the duration of the
        simulation is N * dt (in ms).
    """
    def __init__(self, num_types, num_neurons, num_cart, neuron_start, num_dendrite, num_synapse, pre_neuron, post_neuron, syn_thres, syn_slope, syn_power, syn_saturation, syn_delay, V_rev, dt, V, n, V_1, V_2, V_3, V_4, Tphi, offset, non_input_start):
        
        Process.__init__(self)
        self.num_neurons = num_neurons
        self.dt = dt
        
        delay_steps = int(round(max(syn_delay) * 1e-3 / dt ))
        
        self.buffer = CircularArray(num_neurons, delay_steps, V)
        
        self.neurons = MorrisLecar(num_neurons, num_types, num_cart, neuron_start, dt, num_dendrite, V, n, V_1, V_2, V_3, V_4, Tphi, offset, non_input_start)
        self.synapses = VectorSynapse(num_synapse, pre_neuron, post_neuron, syn_thres, syn_slope, syn_power, syn_saturation, syn_delay, V_rev, dt)
        
        self.st1 = None
        self.st2 = None

    def run_step(self, I_ext, out, put = False):
        
        self.neurons.I_pre.fill(0)
        self.neurons.update_I_pre_input(I_ext)
        
        self.neurons.read_synapse(self.synapses.conductance, self.synapses.V_rev)
        
        self.neurons.eval(self.buffer)
        
        
        self.synapses.compute_synapse(self.buffer)
        
        cuda.memcpy_dtoh(out, self.neurons.V.gpudata)
        self.buffer.step()

class CircularArray:
    def __init__(self, num_neurons, delay_steps, rest):
        self.dtype = np.double
        self.num_neurons = num_neurons
        self.delay_steps = delay_steps
        
        self.buffer = parray.empty((delay_steps, num_neurons),np.double)
        
        d_rest = garray.to_gpu(rest)
        self.current = 0
        
        #initializing V buffer
        for i in range(delay_steps):
            cuda.memcpy_dtod(int(self.buffer.gpudata) + self.buffer.ld * i, d_rest.gpudata, d_rest.nbytes)
        
    def step(self):
        self.current += 1
        if self.current >= self.delay_steps:
            self.current = 0