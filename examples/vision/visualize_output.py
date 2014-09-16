#!/usr/bin/env python

"""
Visualize vision model output.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import neurokernel.LPU.utils.visualizer as vis

V = vis.visualizer()

conf_input = {}
conf_input['type'] = 'image'
conf_input['clim'] = [0, 0.02]
conf_input['ids'] = [range(32*24)]
conf_input['shape'] = [32, 24]
conf_input['trans'] = True

V.add_LPU('data/vision_input.h5', LPU='Vision')
V.add_plot(conf_input, 'input_Vision')

conf_lam_R1 = {}
conf_lam_R1['type'] = 'image'
conf_lam_R1['clim'] = [-0.055,-0.02]
conf_lam_R1['shape'] = [32, 24]
conf_lam_R1['trans'] = True

conf_lam_L1 = {}
conf_lam_L1['type'] = 'image'
conf_lam_L1['clim'] = [-0.053,-0.047]
conf_lam_L1['shape'] = [32, 24]
conf_lam_L1['trans'] = True

V.add_LPU('lamina_output_gpot.h5', './data/lamina.gexf.gz', 'Ret/Lam')
V.add_plot(conf_lam_R1, 'Ret/Lam', 'R1')
V.add_plot(conf_lam_L1, 'Ret/Lam', 'L1')

conf_med = conf_lam_L1.copy()
conf_med['clim'] = [-0.052, -0.050]

V.add_LPU('medulla_output_gpot.h5', './data/medulla.gexf.gz', 'Medulla')
V.add_plot(conf_med, 'Medulla', 'T5a')

V.rows = 2
V.cols = 2
V.fontsize = 22
V.update_interval = 50
V.out_filename = 'vision_output.avi'
V.codec = 'libtheora'
V.dt = 0.0001
V.run()

