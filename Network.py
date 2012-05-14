#!/usr/bin/env python
import numpy as np
from multiprocessing import Process

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
    def __init__(self, dt = 1e-5, N):
        self.dt = dt
        self.N  = N
        self.neuronsK
        self.ml_neurons

    ###########################################################################
    def run(self, I_ext):
        """
        Simulate the network over the entire duration of a specified external
        input current (I_ext).
        """
        
        if len(I_ext) != self.N:
            raise ValueError('input current length does not match that of \
                              the neuron instance')

    ###########################################################################
    def set_morris_lecar(self, N, V0, V1, V2, V3, V4, G_ca, G_k, G_l, V_ca,
                         V_k, V_l, phi, C_m):
        """
        Set Morris Lecar neurons in the network.
        
        Parameters
        ----------
        N : int
            Number of neurons to be added.
        param : dic
            Keys are: phi, G_ca, V1(to 4), V_ca, V_k, V_l, G_k, G_l, C_m and
            the values default values are from from Foundations of Mathematical
            Neuroscience (G.B. Ermentrout et al. 2010)
        """
        
        # Neuron parameters:
        if len(V1) != N or len(V2) != N or len(V3) != N or len(V4) != N or \
           len(G_ca) != N or len(G_k) != N or len(G_l) != N or \
           len(V_ca) != N or len(V_k) != N or len(V_l) != N or \
           len(phi) != N or len(C_m) != N or I_ext.shape[0] != N:
            raise ValueError('Incorrect parameter data structure shape')
        
        self.ml_V3   = V3
        self.ml_V4   = V4
        self.ml_V1   = V1
        self.ml_V2   = V2
        self.ml_G_ca = G_ca
        self.ml_G_k  = G_k
        self.ml_G_l  = G_l
        self.ml_V_ca = V_ca
        self.ml_V_k  = V_k
        self.ml_V_l  = V_l
        self.ml_phi  = phi
        self.ml_C_m  = C_m
        
        # Initial state
        self.ml_V = V0
        
        # External current:
        self.ml_I_ext = I_ext
        
        # Aggregate presynaptic current (updated at each iteration):
        self.ml_I_pre = np.zeros(N, np.double)
        
        # Array for storing trace of neuron states:
        self.ml_V_hist = np.empty((N, self.Nt), np.double)
