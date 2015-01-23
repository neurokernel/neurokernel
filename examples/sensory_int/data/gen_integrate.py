#!/usr/bin/env python

"""
Generate GEXF configuration of sensory integration LPU.
"""

import networkx as nx

def create_lpu(file_name, lpu_name, N_neu):
    """
    Create a LPU for sensory integration.

    Creates a GEXF file containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The GEXF
    file also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are either spiking or graded potential neurons;
    the Leaky Integrate-and-Fire model is used for the former, while the
    Morris-Lecar model is used for the latter (i.e., the neuron's membrane
    potential is deemed to be its output rather than the time when it emits an
    action potential). Synapses use either the alpha function model or a
    conductance-based model.

    Parameters
    ----------
    file_name : str
        Output GEXF file name.
    lpu_name : str
        Name of the sensory integration LPU.
    N_neu : int
        Number of neurons.
    """

    G = nx.DiGraph()
    G.add_nodes_from(range(N_neu))

    # setup neurons
    for i in xrange(N_neu):
        G.node[i] = {
            'model': 'LeakyIAF',
            'name': 'int_%d' % i,
            'extern': True,
            'public': True,
            'spiking': True,
            'V': 0.0,
            'Vr': 0.0,
            'Vt': 0.02,
            'R': 2.0,
            'C': 0.01}

    nx.write_gexf(G, file_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='integrate.gexf.gz',
                        help='LPU file name')
    parser.add_argument('-n', '--num', type=int, default=8,
                        help='Number of neurons')
    parser.add_argument('-l', '--lpu', type=str, default='int',
                        help='LPU name')

    args = parser.parse_args()

    create_lpu(args.lpu_file_name, args.lpu, args.num)
