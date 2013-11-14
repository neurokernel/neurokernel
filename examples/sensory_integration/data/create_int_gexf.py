#!/usr/bin/env python

"""
Create GEXF file detailing integration module neurons.
"""

import networkx as nx

N = 10
g = nx.MultiDiGraph()
g.add_nodes_from(range(N))

for i in xrange(N):
    g.node[i] = {'type': 'LeakyIAF',
                 'name': 'int_%i' % i,
                 'Vr': -0.07,
                 'Vt': -0.025,
                 'R': 1.0,
                 'C': 0.07,
                 'spiking': True,
                 'public': True,
                 'input': True}

nx.write_gexf(g, 'integrate.gexf.gz')

