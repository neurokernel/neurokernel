#!/usr/bin/env python

"""
Visualize vision model output.
"""

import numpy as np
import neurokernel.LPU.utils.visualizer as vis

V = vis.visualizer()

config1 = {}
config1['type'] = 'image'
config1['shape'] = [32,24]
config1['clim'] = [-0.55,-0.2]
config2 = config1.copy()
config2['clim'] = [-0.52,-0.51]

V.add_LPU('lamina_output_gpot.h5', './data/lamina.gexf.gz','lamina')
V.add_plot(config1, 'lamina', 'R1')
V.add_plot(config2, 'lamina', 'L1')

V.add_LPU('medulla_output_gpot.h5', './data/medulla.gexf.gz','medulla')
V.add_plot(config2, 'medulla', 'T5a')


V._update_interval = 50
V.out_filename = 'vision_output.avi'
V.codec = 'libtheora'
V.dt = 0.0001
V.run()

