#!/usr/bin/env python

"""
Visualize vision model output.
"""

import numpy as np
import neurokernel.LPU.utils.visualizer as vis

V = vis.visualizer()

conf_input = {}
conf_input['type'] = 'image'
conf_input['clim'] = [0, 0.5]
conf_input['ids'] = [range(32*24)]
conf_input['shape'] = [32, 24]

V.add_LPU('data/vision_input.h5')
V.add_plot(conf_input, 'input')

conf_lam_R1 = {}
conf_lam_R1['type'] = 'image'
conf_lam_R1['clim'] = [-0.55,-0.2]
conf_lam_R1['shape'] = [32, 24]

conf_lam_L1 = {}
conf_lam_L1['type'] = 'image'
conf_lam_L1['clim'] = [-0.52,-0.51]
conf_lam_L1['shape'] = [32, 24]

V.add_LPU('lamina_output_gpot.h5', './data/lamina.gexf.gz', 'lamina')
V.add_plot(conf_lam_R1, 'lamina', 'R1')
V.add_plot(conf_lam_L1, 'lamina', 'L1')

conf_med = conf_lam_L1.copy()

V.add_LPU('medulla_output_gpot.h5', './data/medulla.gexf.gz', 'medulla')
V.add_plot(conf_med, 'medulla', 'T5a')

V.rows = 1
V.cols = 4
V.update_interval = 50
V.out_filename = 'vision_output.avi'
V.codec = 'libtheora'
V.dt = 0.0001
V.run()

