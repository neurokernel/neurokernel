#!/usr/bin/env python

"""
Visualize demo of artificial LPU integration output.

Notes
-----
Generate demo output by running

python integration_demo.py
"""

import futures

import numpy as np
import matplotlib as mpl
mpl.use('agg')

import neurokernel.LPU.utils.visualizer as vis
import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                        'true':True, 'True':True}

def run(i, out_name):
    G = nx.read_gexf('./data/artificial_lpu_%s.gexf.gz' % i)
    neu_out = [k for k, n in G.node.items() if n['name'][:3] == 'out']

    V = vis.visualizer()

    V.add_LPU('./data/artificial_input_%s.h5' % i, LPU='Sensory')
    V.add_plot({'type':'waveform', 'ids':[[0]]}, 'input_Sensory')

    V.add_LPU('artificial_output_%s_%s_spike.h5' % (i, out_name),
              './data/artificial_lpu_%s.gexf.gz' % i, 'Artificial LPU')    
    V.add_plot({'type': 'raster',
                'ids': {0: range(len(neu_out))},
                'yticks': range(1, 1+len(neu_out)),
                'yticklabels': range(len(neu_out))},
                'Artificial LPU','Output')

    V._update_interval = 50
    V.rows = 2
    V.cols = 1
    V.fontsize = 18
    V.out_filename = 'artificial_output_%s_%s.avi' % (i, out_name)
    V.codec = 'libtheora'
    V.dt = 0.0001
    V.xlim = [0, 1.0]
    V.run()

with futures.ProcessPoolExecutor() as executor:
    for i in [0, 1]:
        for out_name in ['un', 'co']:
            print i, out_name
            executor.submit(run, i, out_name)
