#!/usr/bin/env python

"""
Create GEXF files for testing loading of inter-module connectivity.
"""

import networkx as nx
import neurokernel.core as core
import neurokernel.tools.graph as graph

# Create files containing module connectivity info:
N_A_gpot = 4
N_A_spike = 3
g = nx.MultiDiGraph()
for i in xrange(N_A_gpot+N_A_spike):
    g.add_node('%s' % i)
for i in xrange(0, N_A_gpot):
    g.node['%s' % i]['neuron_type'] = 'gpot'
for i in xrange(N_A_gpot, N_A_gpot+N_A_spike):
    g.node['%s' % i]['neuron_type'] = 'spike'
nx.write_gexf(g, 'A.gexf')

N_B_gpot = 3
N_B_spike = 2
g = nx.MultiDiGraph()
for i in xrange(N_B_gpot+N_B_spike):
    g.add_node('%s' % i)
for i in xrange(0, N_B_gpot):
    g.node['%s' % i]['neuron_type'] = 'gpot'
for i in xrange(N_B_gpot, N_B_gpot+N_B_spike):                
    g.node['%s' % i]['neuron_type'] = 'spike'
nx.write_gexf(g, 'B.gexf')

# Create files containing inter-module connectivity info:
c = core.Connectivity(N_A_gpot, N_A_spike, N_B_gpot, N_B_spike)                     
g = graph.conn_to_graph(c)
nx.write_gexf(g, 'A_B.gexf')
