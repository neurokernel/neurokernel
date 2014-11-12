#!/usr/bin/env python

"""
Visualize vision model output.
"""

import sys

import numpy as np
import matplotlib as mpl

# suppress a warning when running ipython notebook
if 'matplotlib.pyplot' not in sys.modules:
    mpl.use('Agg')

import neurokernel.LPU.utils.visualizer as vis
import networkx as nx
nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                        'true':True, 'True':True}

V = vis.visualizer()

ncols = 32
nrows = 24

conf_input = {}
conf_input['type'] = 'image'
conf_input['clim'] = [0, 0.02]
conf_input['ids'] = [range(nrows*ncols)]
conf_input['shape'] = [nrows, ncols]

V.add_LPU('data/vision_input.h5', LPU='Vision')
V.add_plot(conf_input, 'input_Vision')

conf_lam_R1 = {}
conf_lam_R1['type'] = 'image'
conf_lam_R1['clim'] = [-0.055,-0.02]
conf_lam_R1['shape'] = [nrows, ncols]

conf_lam_L1 = {}
conf_lam_L1['type'] = 'image'
conf_lam_L1['clim'] = [-0.053,-0.047]
conf_lam_L1['shape'] = [nrows, ncols]

conf_lam_L2 = {}
conf_lam_L2['type'] = 'image'
conf_lam_L2['clim'] = [-0.053,-0.047]
conf_lam_L2['shape'] = [nrows, ncols]

conf_lam_L3 = {}
conf_lam_L3['type'] = 'image'
conf_lam_L3['clim'] = [-0.052,-0.048]
conf_lam_L3['shape'] = [nrows, ncols]

conf_lam_L4 = {}
conf_lam_L4['type'] = 'image'
conf_lam_L4['clim'] = [-0.052,-0.048]
conf_lam_L4['shape'] = [nrows, ncols]

V.add_LPU('lamina_output_gpot.h5', './data/lamina.gexf.gz', 'Ret/Lam')
V.add_plot(conf_lam_R1, 'Ret/Lam', 'R2')
V.add_plot(conf_lam_L1, 'Ret/Lam', 'L1')
V.add_plot(conf_lam_L2, 'Ret/Lam', 'L2')
V.add_plot(conf_lam_L3, 'Ret/Lam', 'L3')
V.add_plot(conf_lam_L4, 'Ret/Lam', 'L4')

V.rows = 3
V.cols = 2
V.fontsize = 22
V.update_interval = 50
V.out_filename = 'vision_output.avi'
V.codec = 'libtheora'
V.dt = 0.0001
V.run()

