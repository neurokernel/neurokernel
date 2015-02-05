#!/usr/bin/env python

"""
Generate GEXF configuration of sensory integration LPU.
"""

import networkx as nx

def create_lpu(file_name, lpu_name, N_neu):
    """
    Create a LPU for sensory integration.

    Creates a GEXF file containing the neuron, synapse and port parameters for
    an LPU containing the specified number of neurons. The LPU receives input
    from two LPUs, the antennal lobe and the medulla.

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
            'selector': '/%s/out/spk%d' % (lpu_name, i),
            'extern': False,
            'public': True,
            'spiking': True,
            'V': 0.0,
            'Vr': 0.0,
            'Vt': 0.02,
            'R': 2.0,
            'C': 0.01}

    # setup gpot ports
    for i in xrange(N_neu):
        idx = i + N_neu
        G.add_node(idx, {
            'model': 'port_in_gpot',
            'name': 'port_in_gpot_%d' % i,
            'selector': '/%s/in/gpot%d' % (lpu_name, i),
            'public': False,
            'extern': False,
            'spiking': False })
        # connect gpot ports to neurons via power_gpot_gpot_synapse
        G.add_edge(idx, i, type='directed', attr_dict={
            'name': G.node[idx]['name']+'-'+G.node[i]['name'],
            'model': 'power_gpot_gpot',
            'class': 2,
            'conductance': True,
            'slope': 4e9,
            'reverse': -0.015,
            'saturation': 30,
            'power': 4.0,
            'delay': 1.0,
            'threshold': -0.061})

    # setup spiking ports
    for i in xrange(3):
        idx = i + 2*N_neu
        G.add_node(idx, {
            'model': 'port_in_spk',
            'name': 'port_in_spk_%d' % i,
            'selector': '/%s/in/spk%d' % (lpu_name, i),
            'public': False,
            'extern': False,
            'spiking': True })
        # connect spiking ports to neurons
        for j in xrange(N_neu):
            G.add_edge(idx, j, type='directed', attr_dict={
                'name': G.node[idx]['name']+'-'+G.node[j]['name'],
                'model': 'AlphaSynapse',
                'class': 0,
                'conductance': True,
                'ad': 0.19*1000,
                'ar': 1.1*100,
                'gmax': 0.003,
                'reverse': 0.065})

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
