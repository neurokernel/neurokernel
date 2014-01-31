#!/usr/bin/env python

"""
LPU parser that parsers local processing unit (LPU) specification of extended
graphics xml format (EGXF) into python data structure. The python package
NetworkX is used for generating and storing graphic representation of LPU.

- lpu_parser            - GEXF-to-python LPU parser.
"""

__all__ = ['lpu_parser']

import networkx as nx

# Work around bug that causes networkx to choke on GEXF files with boolean
# attributes that contain the strings 'True' or 'False'
# (bug already observed in https://github.com/networkx/networkx/pull/971)
nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                        'true':True, 'True':True}

import numpy
from collections import defaultdict


def lpu_parser(filename):
    """
    GEXF-to-python LPU parser.

    Convert a .gexf LPU specifications into NetworkX graph type, and
    then pack data into list of dictionaries to be passed to the LPU
    module. The individual parameters will be represented by lists.

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
    2. Each node should have an integer attribute called 'model' indicating
       the model to be used for that neuron( Eg:- IAF, Morris-Lecar).
       Refer the documentation of LPU.neurons.BaseNeuron to implement
       custom neuron models.
    3. The attributes of the nodes should be consistent across all nodes
       of the same model. For example if a particular node of model 'IAF'
       has attribute 'bias', all nodes of model 'IAF' must necessarily
       have this attribute.
    4. Each node should have an boolean attribute called public - indicating
       whether that neuron either recieves input or provides output to
       other LPUs.
    5. Each node should have an boolean attribute called extern indicating
       whether the neuron accepts external input from a file.
    6. Each edge(synapse) in the graph should have an integer
       atribute called 'class' which should be one of the following values.
          0. spike-spike synapse
          1. spike-gpot synapse
          2. gpot-spike synapse
          3. gpot-gpot synapse
    7. Each edge should have an integer attribute called 'model' indicating
       the model to be used for that synapse( Eg:- alpha).
       Refer the documentation of LPU.synapses.BaseSynapse to implement
       custom synapse models.
    8. The attributes of the nodes should be consistent across all nodes
       of the same model.
    9. Each edge should have a boolean attribute called 'conductance'
       representing whether it's output is a conductance or current.
    10.For all edges with the 'conductance' attribute true, there should
       be an attribute called 'reverse'
    """


    '''
    Need to add code to assert all conditions mentioned above are met
    '''
    graph = nx.read_gexf(filename)
    models = []
    n_dict_list = []
    neurons = graph.node
    if len(neurons) > 0:
        for i in range(len(neurons)):
            if not str(neurons[str(i)]['model']) in models:
                n_dict = dict.fromkeys(neurons[str(i)])
                for key in n_dict.iterkeys():
                    n_dict[key] = list()
                n_dict['id'] = list()
                n_dict_list.append(n_dict)
                models.append(str(neurons[str(i)]['model']))
            ind = models.index(str(neurons[str(i)]['model']))
            for key in neurons[str(i)].iterkeys():
                n_dict_list[ind][key].append(neurons[str(i)][key])
            n_dict_list[ind]['id'].append(i)
    else:
        n_dict_list = None

    synapses = graph.edges(data=True)
    models = []
    s_dict_list = []
    synapses.sort(cmp=synapse_cmp)
    if len(synapses) > 0:


        for i in range(len(synapses)):
            if not str(synapses[i][2]['model']) in models:
                s_dict = dict.fromkeys(synapses[i][2])
                for key in s_dict.viewkeys():
                    s_dict[key] = list()
                s_dict['post'] = list()
                s_dict['pre'] = list()
                s_dict_list.append(s_dict)
                models.append(str(synapses[i][2]['model']))
            ind = models.index(str(synapses[i][2]['model']))
            s_dict_list[ind]['pre'].append(synapses[i][0])
            s_dict_list[ind]['post'].append(synapses[i][1])
            for key in synapses[i][2].viewkeys():
                s_dict_list[ind][key].append(synapses[i][2][key])
    else:
        s_dict_list = None

    return n_dict_list, s_dict_list


def synapse_cmp(x, y):
    if int(x[1]) < int(y[1]):
        return -1
    elif int(x[1]) > int(y[1]):
        return 1
    else:
        return 0
