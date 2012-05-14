#!/usr/bin/env python
import numpy as np
from multiprocessing import Process
import time
import parray

"""
Can you please construct an object-oriented PyCUDA implementation of a network of ML neurons randomly connected by alpha function synapses (similar to the IAF network you implemented for E9070) that we can use for testing the architecture?
I imagine that a standard numerical ODE solver such as low-order Runge-Kutta should be sufficient for simulating the network.
Also, I'm not sure what dynamics we should expect to observe for such a network. You may need to talk to Yiyin or Nikul for further information about both of these points.
"""

# _   _      _                      _    
#| \ | | ___| |___      _____  _ __| | __
#|  \| |/ _ \ __\ \ /\ / / _ \| '__| |/ /
#| |\  |  __/ |_ \ V  V / (_) | |  |   < 
#|_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\
#
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
    def __init__(self, dt = 1e-5, N = 0):
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
