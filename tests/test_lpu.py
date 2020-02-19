#!/usr/bin/env python

from unittest import main, TestCase

import networkx as nx

from neurokernel.LPU.LPU import LPU

class test_lpu(TestCase):
   def test_graph_to_dicts(self):
       self.maxDiff = 2048
       g = nx.MultiDiGraph()
       g.add_node('0', **{'model': 'LeakyIAF',
                        'spiking': True,
                        'extern': True,
                        'public': True,
                        'selector': '/lif0',
                        'C': 1.0,
                        'R': 1.0,
                        'V': -1.0,
                        'Vr': -0.1,
                        'Vt': -0.1,
                        'name': 'lif0'})
       g.add_node('1', **{'model': 'LeakyIAF',
                        'spiking': True,
                        'extern': True,
                        'public': True,
                        'selector': '/lif1',
                        'C': 2.0,
                        'R': 2.0,
                        'V': -2.0,
                        'Vr': -0.2,
                        'Vt': -0.2,
                        'name': 'lif1'})
       g.add_node('2', **{'model': 'MorrisLecar',
                        'spiking': False,
                        'extern': False,
                        'public': False,
                        'selector': '/ml0',
                        'V1': 0.03,
                        'V2': 0.3,
                        'V3': 0.2,
                        'V4': 0.1,
                        'initV': -0.1,
                        'initn': 0.1,
                        'offset': 0,
                        'phi': 0.01,
                        'name': 'ml0'})
       g.add_node('3', **{'model': 'MorrisLecar',
                        'spiking': False,
                        'extern': False,
                        'public': False,
                        'selector': '/ml1',
                        'V1': 0.04,
                        'V2': 0.4,
                        'V3': 0.3,
                        'V4': 0.2,
                        'initV': -0.2,
                        'initn': 0.2,
                        'offset': 0,
                        'phi': 0.02,
                        'name': 'ml1'})
       g.add_edge('0', '1', **{'class': 0,
                                       'model': 'AlphaSynapse',
                                       'conductance': True,
                                       'name': 'lif0-lif1',
                                       'reverse': 0.01,
                                       'ad': 0.01,
                                       'gr': 1.0,
                                       'gmax': 0.001})
       g.add_edge('1', '0', **{'class': 0,
                                       'model': 'AlphaSynapse',
                                       'conductance': True,
                                       'name': 'lif1-lif0',
                                       'reverse': 0.02,
                                       'ad': 0.02,
                                       'gr': 2.0,
                                       'gmax': 0.002})

       n_dict, s_dict = LPU.graph_to_dicts(g)
       self.assertDictEqual(n_dict,
                            {'LeakyIAF':
                             {'C': [1.0, 2.0],
                              'name': ['lif0', 'lif1'],
                              'id': [0, 1],
                              'selector': ['/lif0', '/lif1'],
                              'Vr': [-0.1, -0.2],
                              'R': [1.0, 2.0],
                              'Vt': [-0.1, -0.2],
                              'V': [-1.0, -2.0],
                              'extern': [True, True],
                              'spiking': [True, True],
                              'public': [True, True]},
                             'MorrisLecar': {
                                 'V1': [0.03,0.04],
                                 'V2': [0.3, 0.4],
                                 'V3': [0.2, 0.3],
                                 'V4': [0.1, 0.2],
                                 'initV': [-0.1, -0.2],
                                 'initn': [0.1, 0.2],
                                 'offset': [0, 0],
                                 'phi': [0.01, 0.02],
                                 'selector': ['/ml0','/ml1'],
                                 'name': ['ml0','ml1'],
                                 'id': [2, 3],
                                 'extern': [False, False],
                                 'spiking': [False, False],
                                 'public': [False, False]}})

       self.assertDictEqual(s_dict,
                            {'AlphaSynapse':
                             {'pre': ['1', '0'],
                              'reverse': [0.02, 0.01],
                              'gmax': [0.002, 0.001],
                              'post': ['0', '1'],
                              'class': [0, 0],
                              'conductance': [True, True],
                              'ad': [0.02, 0.01],
                              'gr': [2.0, 1.0],
                              'id': [0, 1],
                              'name': ['lif1-lif0', 'lif0-lif1']}})

if __name__ == '__main__':
    main()
