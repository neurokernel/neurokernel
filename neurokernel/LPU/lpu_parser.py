#!/usr/bin/env python

"""
LPU parser that parsers local processing unit (LPU) specification of extended 
graphics xml format (EGXF) into python data structure. The python package
NetworkX is used for generating and storing graphic representation of LPU.

- lpu_parser            - GEXF-to-python LPU parser.
"""

__all__ = ['lpu_parser']

__author__ = """\n""".join(['Nikul Ukani <nhu2001@columbia.edu>',
                            'Chung-Heng Yeh <chyeh@ee.columbia.edu>',
                            'Yiyin Zhou <yz2227@columbia.edu>'])

import networkx as nx
import numpy
from collections import defaultdict


def lpu_parser(filename):
    """
    GEXF-to-python LPU parser.

    Convert a .gexf LPU specifications into NetworkX graph type, and 
    then pack data into list of dictionaries to be passed to the LPU
    module.
    
    Parameters
    ----------
    filename : String
        Filename containing LPU specification of GEXF format.
        See Notes for requirements to be met by the GEXF file.

    Returns
    -------
    n_dict : list of dictionaries
    
    s_dict : list of dictionaries
    
    Notes
    -----

    1. Each node(neuron) in the graph should necessarily have
       a boolean attribute called 'spiking' indicating whether the neuron is
       spiking or graded potential.
    2. Each node should have an integer attribute called 'type' indicating
       the model to be used for that neuron( Eg:- IAF, Morris-Lecar).
       Refer the documentation of LPU.neurons.BaseNeuron to implement
       custom neuron models.
    3. The attributes of the nodes should be consistent across all nodes
       of the same type. For example if a particular node of type 'IAF'
       has attribute 'bias', all nodes of type 'IAF' must necessarily
       have this attribute.
    4. Each node should have an boolean attribute called public - indicating
       whether that neuron either recieves input or provides output to
       other LPUs.
    5. Each node should have an boolean attribute called input indicating
       whether the neuron accepts external input from a file.
    6. Each edge(synapse) in the graph should have an integer
       atribute called 'class' which should be one of the following values.
          0. spike-spike synapse
          1. spike-gpot synapse
          2. gpot-spike synapse
          3. gpot-gpot synapse
    7. Each edge should have an integer attribute called 'type' indicating
       the model to be used for that synapse( Eg:- alpha).
       Refer the documentation of LPU.synapses.BaseSynapse to implement
       custom synapse models.
    8. The attributes of the nodes should be consistent across all nodes
       of the same type.
    9. Each edge should have a boolean attribute called 'conductance'
       representing whether it's output is a conductance or current.
    
    """


    '''
    Parse the file using networkx and pack the data into n_dict and s_dict.
    Assert if all the conditions mentioned above are met. If not, log an
    error.
    Assert if the synapse class matches the neurons they connect.
    Return n_dict and s_dict
    '''
