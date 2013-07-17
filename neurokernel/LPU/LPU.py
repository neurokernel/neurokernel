#!/usr/bin/env python

import numpy as np
import parray
from neurokernel.core import Module
import neurokernel.base as base
from types import *

class LPU(Module):
    def __init__(self, dt, n_dict, s_dict, input_file=None,
                 device=None, output_file=None, port_ctrl=base.PORT_CTRL,
                 port_data=base.PORT_DATA, LPU_id=None, debug=False):

    """
    Initialization of a LPU

    Parameters
    ----------
    
    dt : double
        one time step.
    n_dict : list of dictionaries
        a list of dictionaries describing the neurons in this LPU - one
        dictionary for each type of neuron.
    s_dict : list of dictionaries
        a list of dictionaries describing the synapses in this LPU - one
        for each type of synapse.
    input_file : string
        path and name of the input video file
    output_file : string
        path and name of the output files
    port_data : int
        Port to use when communicating with broker.
    port_ctrl : int
        Port used by broker to control module.
    device : int
         Device no to use
    LPU_id : string
        Name of the LPU
    debug : boolean
         This parameter will be passed to all the neuron and synapse
         objects instantiated by this LPU and is intended to be used
         for debugging purposes only.
         Will be set to False by default

    """

    super(LPU, self).__init__(port_data=port_data, port_ctrl=port_ctrl,
                              device=device, id=LPU_id)


    self.dt = dt
    self.debug = debug
    self.device = device
    self.n_dict = n_dict
    self.s_dict = s_dict
    self.output_file = self.output_file
    self.input_file = self.input_file


    def pre_run():
        super(LPU,self).pre_run()
        self.setup_connectivity()
        self.init_gpu()
        self.init_objects()
        
    def post_run():
        super(LPU,self).post_run()
        
        #Call the post run methods of all neuron and synapse objects.

    def setup_connectivity():
        '''
        Parse _conn_dict, n_dict, s_dict and create dummy neurons
        representing inputs from other LPUs.
        '''

    def init_gpu():
        '''
        Create CUDA context.
        '''
        
    def init_objects():
        '''
        Instantiate neuron and synapse objects for each type.
        Instantiate buffer objects for all synapse types to support delay.
        Create dictionaries cond and non_cond representing the
        input from synapses to each neuron type.
        '''

    def run_step(in_gpot_dict, in_spike_dict, out_gpot, out_spike):
        super(LPU, self).run_step()

        '''
        Process in_gpot_dict and update dummy neurons.
        Update dummy neurons for all neuron types.
        '''

        for each neuron in self.neurons:
            # Read input from external file
            neuron.update_I_external()
            # Compute current from conductance based synapses
            neuron.update_I_cond(cond[neuron.__type__].gpudata)
            # Compute current from non-conductance based synapses
            neuron.update_I(