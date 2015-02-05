import matplotlib as mpl
mpl.use('Agg')

import neurokernel.core as core
import neurokernel.base as base
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.comm import get_random_port

from data.eyeimpl import EyeGeomImpl
from matplotlib.colors import Normalize

import networkx as nx
nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}
import numpy as np
import neurokernel.LPU.utils.visualizer as vis

RET_GEXF_FILE = 'retina.gexf.gz'
LAM_GEXF_FILE = 'lamina.gexf.gz'

INPUT_FILE = 'vision_input.h5'
IMAGE_FILE = 'image1.mat'
RET_OUTPUT_FILE = 'retina_output'
LAM_OUTPUT_FILE = 'lamina_output'
RET_OUTPUT_GPOT = RET_OUTPUT_FILE + '_gpot.h5'
LAM_OUTPUT_GPOT = LAM_OUTPUT_FILE + '_gpot.h5'
RET_OUTPUT_PNG = 'retina_output.png'
LAM_OUTPUT_PNG = 'lamina_output.png'
RET_OUTPUT_AVI = 'retina_output.avi'
LAM_OUTPUT_AVI = 'lamina_output.avi'
RET_OUTPUT_MPEG = 'retina_output.mp4'
LAM_OUTPUT_MPEG = 'lamina_output.mp4'

V = vis.visualizer()

n = Normalize(vmin=0, vmax=2000, clip=True)
conf_input = {}
conf_input['norm'] = n
conf_input['type'] = 'dome'
V.add_LPU('intensities.h5','retina.gexf.gz', LPU='Vision', is_input=True)
V.add_plot(conf_input, 'input_Vision')
conf_R1 = {}
conf_R1['type'] = 'dome'
V.add_LPU(RET_OUTPUT_GPOT,'retina.gexf.gz', LPU='Retina')
V.add_plot(conf_R1, 'Retina', 'R1')
conf_L1 = {}
conf_L1['type'] = 'dome'
V.add_LPU(LAM_OUTPUT_GPOT, 'lamina.gexf.gz', LPU='Lamina')
V.add_plot(conf_L1, 'Lamina', 'L1')
V.fontsize = 22
V.fps = 5
V.update_interval = 50
V.out_filename = 'vision_output.mp4'
V.codec = 'mpeg4'
V.dt = 0.0001
V.FFMpeg = True      # change to False to use LibAV fork instead( default on UBUNTU)
V.run()
