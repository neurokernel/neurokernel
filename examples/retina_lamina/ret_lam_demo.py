#!/usr/bin/env python

import argparse
import time

import matplotlib as mpl
mpl.use('Agg')

import neurokernel.core as core
import neurokernel.base as base
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.comm import get_random_port

from data.eyeimpl import EyeGeomImpl

import networkx as nx
nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}
import numpy as np
import neurokernel.LPU.utils.visualizer as vis

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rec-micro', dest='record_microvilli',
                    action="store_true", help='records microvilli if set')
parser.add_argument('-u', '--unrec-neuro', dest='record_neuron',
                    action="store_false",
                    help='does not record neuron if set')

parser.add_argument('-l', '--layers', dest='num_layers', type=int,
                    default=16,
                    help='number of layers of ommatidia on circle')
parser.add_argument('-m', '--micro', dest='num_microvilli', type=int,
                    default=30000,
                    help='number of microvilli in each photoreceptor')

parser.add_argument('-d', '--port_data', default=None, type=int,
                    help='Data port [default: randomly selected]')
parser.add_argument('-c', '--port_ctrl', default=None, type=int,
                    help='Control port [default: randomly selected]')
parser.add_argument('-a', '--ret_dev', default=0, type=int,
                    help='GPU for retina lobe [default: 0]')
parser.add_argument('-b', '--lam_dev', default=1, type=int,
                    help='GPU for lamina lobe [default: 1]')

parser.add_argument('-i', '--input', action="store_true",
                    help='generates input if set')
parser.add_argument('-g', '--gexf', action="store_true",
                    help='generates gexf of LPU if set')
parser.add_argument('-o', '--output', action="store_true",
                    help='generates output if set')

parser.add_argument('-s', '--suppress', action="store_true",
                    help='supresses simulation')
parser.add_argument('-t', '--type', type=str, default='image',
                    help='type of input: image/video, ball/bar/grating')
                    
parser.add_argument('--log', default='file', type=str,
                    help='Log output to screen [file, screen, both, or none;'
                         ' default:none]')

parser.add_argument('--steps', default=10, type=int,
                    help='simulation steps')
                    
parser.add_argument('--retina-only', action="store_true",
                    help='if set only retina simulation takes place')

args = parser.parse_args()

#logging setup
file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = base.setup_logger(file_name, screen)


dt = 1e-4
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

print('Instantiating eye geometry')
eyemodel = EyeGeomImpl(args.num_layers, retina_only=args.retina_only)

if args.input:
    print('Generating input of model')
    _dummy = eyemodel.get_intensities(None,
                                      {'type': args.type,
                                       'steps': args.steps,
                                       'dt': dt, 'output_file': INPUT_FILE})
if args.gexf:
    print('Writing retina lpu')
    eyemodel.write_retina(RET_GEXF_FILE)
    if not args.retina_only:
        print('Writing lamina lpu')
        eyemodel.write_lamina(LAM_GEXF_FILE)

if args.port_data is None and args.port_ctrl is None:
    port_data = get_random_port()
    port_ctrl = get_random_port()
else:
    port_data = args.port_data
    port_ctrl = args.port_ctrl

if not args.suppress:
    man = core.Manager(port_data, port_ctrl)
    man.add_brok()

    print('Parsing retina lpu data')
    n_dict_ret, s_dict_ret = LPU.lpu_parser(RET_GEXF_FILE)
    print('Initializing retina LPU')
    lpu_ret = LPU(dt, n_dict_ret, s_dict_ret,
                  input_file=INPUT_FILE,
                  output_file=RET_OUTPUT_FILE, port_ctrl=port_ctrl,
                  port_data=port_data, device=args.ret_dev, id='retina',
                  debug=False)
    man.add_mod(lpu_ret)

    if not args.retina_only:
        print('Parsing lamina lpu data')
        n_dict_lam, s_dict_lam = LPU.lpu_parser(LAM_GEXF_FILE)
        print('Initializing lamina LPU')
        lpu_lam = LPU(dt, n_dict_lam, s_dict_lam,
                      input_file=None,
                      output_file=LAM_OUTPUT_FILE, port_ctrl=port_ctrl,
                      port_data=port_data, device=args.lam_dev, id='lamina',
                      debug=False)

        man.add_mod(lpu_lam)
        print('Connecting retina and lamina')
        eyemodel.connect_retina_lamina(man, lpu_ret, lpu_lam)

    print('Starting simulation')
    start_time = time.time()
    man.start(steps=args.steps)
    man.stop()
    
    print('Simulation complete: Duration {} seconds'.format(time.time() - 
                                                            start_time))

if args.output:
    '''
    eyemodel.visualise_output(media_file=RET_OUTPUT_MPEG,
                              model_output=RET_OUTPUT_GPOT,
                              config = {'LPU': 'retina',
                                        'type':args.type})
    if not args.retina_only:
        eyemodel.visualise_output(media_file=LAM_OUTPUT_MPEG,
                                  model_output=LAM_OUTPUT_GPOT,
                                  config = {'LPU': 'lamina',
                                            'type':args.type,
                                            'neuron': 'L1'} )
    '''
    V = vis.visualizer()
    conf_R1 = {}
    conf_R1['type'] = 'dome'
    V.add_LPU(RET_OUTPUT_GPOT,'retina.gexf.gz', LPU='Retina')
    V.add_plot(conf_R1, 'Retina', 'R1')
    
    if not args.retina_only:
        conf_L1 = {}
        conf_L1['type'] = 'dome'
        V.add_LPU(LAM_OUTPUT_GPOT, 'lamina.gexf.gz', LPU='Lamina')
        V.add_plot(conf_L1, 'Lamina', 'L1')
    
    V.fontsize = 22
    V.fps = 5
    V.update_interval = 50
    V.out_filename = 'vision_output.avi'
    V.codec = 'mpeg4'
    V.dt = 0.0001
    V.FFMpeg = False      # change to False to use LibAV fork instead( default on UBUNTU)
    V.run()
