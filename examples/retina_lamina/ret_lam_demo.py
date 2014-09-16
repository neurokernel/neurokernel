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

from matplotlib.colors import Normalize
    
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
parser.add_argument('-e', '--med_dev', default=1, type=int,
                    help='GPU for medulla [default: 1]')

parser.add_argument('-i', '--input', action="store_true",
                    help='generates input if set')
parser.add_argument('-g', '--gexf', action="store_true",
                    help='generates gexf of LPU if set')
parser.add_argument('-o', '--output', action="store_true",
                    help='generates output if set')

parser.add_argument('-s', '--suppress', action="store_true",
                    help='supresses simulation')
parser.add_argument('-t', '--type', type=str, default='bar',
                    help='type of input: image/video, ball/bar (default)/grating')
                    
parser.add_argument('--log', default='file', type=str,
                    help='Log output to screen [file, screen, both, or none;'
                         ' default:none]')

parser.add_argument('--steps', default=10, type=int,
                    help='simulation steps')
                    
parser.add_argument('--model', default='r', type=str,
                    help='set the initials of the LPUs to simulate: r(etina)'
                         ' l(amina) m(edulla), default "r", checks for'
                         ' validity will be applied')

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
MED_GEXF_FILE = 'medulla.gexf.gz'

IMAGE_FILE = 'image1.mat'

RET_INPUT = 'vision_input.h5'
#TODO KP depends on Nikul's work
LAM_INPUT = None
MED_INPUT = None
RET_OUTPUT_FILE = 'retina_output'
LAM_OUTPUT_FILE = 'lamina_output'
MED_OUTPUT_FILE = 'medulla_output'
RET_OUTPUT_GPOT = RET_OUTPUT_FILE + '_gpot.h5'
LAM_OUTPUT_GPOT = LAM_OUTPUT_FILE + '_gpot.h5'
MED_OUTPUT_GPOT = MED_OUTPUT_FILE + '_gpot.h5'
RET_OUTPUT_PNG = 'retina_output.png'
LAM_OUTPUT_PNG = 'lamina_output.png'
RET_OUTPUT_AVI = 'retina_output.avi'
LAM_OUTPUT_AVI = 'lamina_output.avi'
RET_OUTPUT_MPEG = 'retina_output.mp4'
LAM_OUTPUT_MPEG = 'lamina_output.mp4'

# XXX eyemodel's calculations that depend on model are checked internally
# that will cause some messages to be printed, like  'Writing retina lpu'
# without that necessarily taking place

print('Instantiating eye geometry')
eyemodel = EyeGeomImpl(args.num_layers, model=args.model)

if args.input:
    print('Generating input of model')

    config = {'type': args.type, 'steps': args.steps,
              'dt': dt, 'output_file': RET_INPUT}
    '''
    replace with above for bar generation
    config = {'type': 'bar', 'steps': args.steps,
              'dt': dt, 'shape': (100,100),
              'width': 20, 'speed': 100, 'dir':0}
    '''
    _dummy = eyemodel.get_intensities(file=None, config=config)

if args.gexf:
    print('Writing retina lpu')
    eyemodel.write_retina(RET_GEXF_FILE)
    print('Writing lamina lpu')
    eyemodel.write_lamina(LAM_GEXF_FILE)
    print('Writing medulla lpu')
    eyemodel.write_medulla(MED_GEXF_FILE)

if args.port_data is None and args.port_ctrl is None:
    port_data = get_random_port()
    port_ctrl = get_random_port()
else:
    port_data = args.port_data
    port_ctrl = args.port_ctrl

if not args.suppress:
    man = core.Manager(port_data, port_ctrl)
    man.add_brok()

    if 'r' in args.model:
        print('Parsing retina LPU data')
        n_dict_ret, s_dict_ret = LPU.lpu_parser(RET_GEXF_FILE)
        print('Initializing retina LPU')
        lpu_ret = LPU(dt, n_dict_ret, s_dict_ret,
                      input_file=RET_INPUT,
                      output_file=RET_OUTPUT_FILE, port_ctrl=port_ctrl,
                      port_data=port_data, device=args.ret_dev, id='retina',
                      debug=True)
        man.add_mod(lpu_ret)

    if 'l' in args.model:
        print('Parsing lamina LPU data')
        n_dict_lam, s_dict_lam = LPU.lpu_parser(LAM_GEXF_FILE)
        print('Initializing lamina LPU')
        lpu_lam = LPU(dt, n_dict_lam, s_dict_lam,
                      input_file=LAMINA_INPUT,
                      output_file=LAM_OUTPUT_FILE, port_ctrl=port_ctrl,
                      port_data=port_data, device=args.lam_dev, id='lamina',
                      debug=True)
        man.add_mod(lpu_lam)

    if 'm' in args.model:
        print('Parsing medulla LPU data')
        n_dict_med, s_dict_med = LPU.lpu_parser(MED_GEXF_FILE)
        print('Initializing medulla LPU')
        lpu_med = LPU(dt, n_dict_med, s_dict_med,
                      input_file=MEDULLA_INPUT,
                      output_file=MED_OUTPUT_FILE, port_ctrl=port_ctrl,
                      port_data=port_data, device=args.med_dev, id='medulla',
                      debug=True)
        man.add_mod(lpu_med)

    # if gexf files are not written again, 
    # patterns are loaded from previous files too
    from_file = not args.gexf
    print('Connecting retina and lamina')
    eyemodel.connect_retina_lamina(man, lpu_ret, lpu_lam, from_file)
    print('Connecting lamina and medulla')
    eyemodel.connect_lamina_medulla(man, lpu_lam, lpu_med, from_file)

    print('Starting simulation')
    start_time = time.time()
    man.start(steps=args.steps)
    man.stop()
    
    print('Simulation complete: Duration {} seconds'.format(time.time() - 
                                                            start_time))

if args.output:

    V = vis.visualizer()

    n = Normalize(vmin=0, vmax=30, clip=True)
    conf_input = {}
    conf_input['norm'] = n
    conf_input['type'] = 'dome'
    V.add_LPU('intensities.h5','retina.gexf.gz', LPU='Vision', is_input=True)
    V.add_plot(conf_input, 'input_Vision')
    
    n1 = Normalize(vmin=-70, vmax=0, clip=True)    
    conf_R1 = {}
    conf_R1['type'] = 'dome'
    conf_R1['norm'] = n1
    V.add_LPU(RET_OUTPUT_GPOT,'retina.gexf.gz', LPU='Retina')
    V.add_plot(conf_R1, 'Retina', 'R1')
    
    if not args.retina_only:
        conf_L1 = {}
        conf_L1['type'] = 'dome'
        V.add_LPU(LAM_OUTPUT_GPOT, 'lamina.gexf.gz', LPU='Lamina')
        V.add_plot(conf_L1, 'Lamina', 'L1')

        conf_T4a = {}
        conf_T4a['type'] = 'dome'
        V.add_LPU(MED_OUTPUT_GPOT, 'medulla.gexf.gz', LPU='Medulla')
        V.add_plot(conf_T4a, 'Medulla', 'T4a')
        
    V.fontsize = 22
    V.fps = 5
    V.update_interval = 50
    V.out_filename = 'vision_output.mp4'
    V.codec = 'mpeg4'
    V.dt = 0.0001
    #V.FFMpeg = True      # change to False to use LibAV fork instead( default on UBUNTU)
    V.run()
