#!/usr/bin/env python

"""
Demonstration of how to implement a Morris-Lecar neuron.
"""

import numpy as np
import scipy as sp
import scipy.integrate

# Parameters used by Yiyin and Nikul:
params = {'phi': 0.01,
          'G_ca': 1.1,
          'V3': -0.4,
          'V4': 0.1,
          'V_ca': 1.0,
          'V_k': -0.7,
          'V_l': -0.5,
          'G_k': 2.0,
          'G_l': 0.5,
          'V1': -1.2,
          'V2': 0.15,
          'C_m': 1.0}

# Other parameters from Foundations of Mathematical Neuroscience (G.B.
# Ermentrout et al. 2010):
params_hopf = {'phi': 0.04,
               'G_ca': 4.4,
               'V3': 2.0,
               'V4': 30.0,
               'V_ca': 120.0,
               'V_k': -84.0,
               'V_l': -60.0,
               'G_k': 8.0,
               'G_l': 2.0,
               'V1': -1.2,
               'V2': 18.0,
               'C_m': 20.0}

def morris_lecar_neuron(t, I_func, params=params):
    """
    Compute the response of a Morris-Lecar neuron with the specified parameters
    and external input current function over the specified range of times.
    """

    V3 = params['V3']
    V4 = params['V4']
    V1 = params['V1']
    V2 = params['V2']
    G_ca = params['G_ca']
    G_k = params['G_k']
    G_l = params['G_l']
    V_ca = params['V_ca']
    V_k = params['V_k']
    V_l = params['V_l']
    phi = params['phi']
    C_m = params['C_m']

    def f(x, t):
        N, V = x

        N_ss = 0.5*(1.0+np.tanh((V-V3)/V4))
        M_ss = 0.5*(1.0+np.tanh((V-V1)/V2))
        tau_N = 1.0/(phi*np.cosh((V-V3)/(2*V4)))

        dN = (N_ss-N)/tau_N        
        dV = (I_func(t)-G_l*(V-V_l)-G_ca*M_ss*(V-V_ca)-G_k*N*(V-V_k))/C_m
        return dN, dV

    return sp.integrate.odeint(f, (1.0, 5.0), t)

if __name__ == '__main__':
    t = np.arange(0, 100.0, 1e-4)

    data = morris_lecar_neuron(t, lambda t: 18.55)
