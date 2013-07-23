"""
Unit tests for neurokernel.tools.graph
"""

from unittest import main, TestCase, TestSuite

import numpy as np
import networkx as nx

import neurokernel.base as base
import neurokernel.core as core
import neurokernel.tools.graph as graph

class test_graph(TestCase):
    def test_graph_to_conn_BaseConnectivity(self):
        g = nx.MultiDiGraph()
        g.add_nodes_from(['A:0', 'A:1', 
                          'B:0', 'B:1', 'B:2'])
        g.add_edges_from([('A:0', 'B:1'), ('A:1', 'B:2')])
        c = graph.graph_to_conn(g, base.BaseConnectivity)
        np.all(np.array([[0, 1, 0],
                         [0, 0, 1]])==c['A', :, 'B', :])

    def test_graph_to_conn_Connectivity(self):
        g = nx.MultiDiGraph()
        g.add_nodes_from(['A:0', 'A:1', 
                          'B:0', 'B:1', 'B:2'])        
        g.node['A:0']['neuron_type'] = 'gpot'
        g.node['A:1']['neuron_type'] = 'spike'        
        g.node['B:0']['neuron_type'] = 'gpot'
        g.node['B:1']['neuron_type'] = 'gpot'
        g.node['B:2']['neuron_type'] = 'spike'                
        g.add_edges_from([('A:0', 'B:1'), ('A:1', 'B:2')])
        c = graph.graph_to_conn(g, core.Connectivity)
        np.all(np.array([[0, 1]])==c['A', 'gpot', :, 'B', 'gpot', :])
        np.all(np.array([[0]])==c['A', 'gpot', :, 'B', 'spike', :])
        np.all(np.array([[0, 0]])==c['A', 'spike', :, 'B', 'gpot', :])
        np.all(np.array([[1]])==c['A', 'spike', :, 'B', 'spike', :])        
        
if __name__ == '__main__':
    main()
