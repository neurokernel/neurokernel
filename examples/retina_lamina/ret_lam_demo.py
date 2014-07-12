#!/usr/bin/env python

import argparse

import neurokernel.core as core
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.comm import get_random_port

from data.eyeimpl import EyeGeomImpl



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
parser.add_argument('-b', '--lam_dev', default=0, type=int,
                    help='GPU for lamina lobe [default: 0]')

parser.add_argument('-i', '--input', action="store_true",
                    help='generates input if set')
parser.add_argument('-g', '--gexf', action="store_true",
                    help='generates gexf of LPU if set')

args = parser.parse_args()

dt = 1e-4
RET_GEXF_FILE = 'retina.gexf.gz'
LAM_GEXF_FILE = 'lamina.gexf.gz'

INPUT_FILE = 'vision_input.h5'
IMAGE_FILE = 'image1.mat'
RET_OUTPUT_FILE = 'retina_output.h5'
LAM_OUTPUT_FILE = 'lamina_output.h5'

eyemodel = EyeGeomImpl(args.num_layers)

if args.input:
    eyemodel.generate_input(IMAGE_FILE, INPUT_FILE)
if args.gexf:
    eyemodel.generate_retina(RET_GEXF_FILE)
    eyemodel.generate_lamina(LAM_GEXF_FILE)
if args.port_data is None and args.port_ctrl is None:
    port_data = get_random_port()
    port_ctrl = get_random_port()
else:
    port_data = args.port_data
    port_ctrl = args.port_ctrl

#TODO
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

print('Parsing lamina lpu data')
n_dict_lam, s_dict_lam = LPU.lpu_parser(LAM_GEXF_FILE)
print('Initializing lamina LPU')
lpu_lam = LPU(dt, n_dict_lam, s_dict_lam,
              input_file=INPUT_FILE,
              output_file=LAM_OUTPUT_FILE, port_ctrl=port_ctrl,
              port_data=port_data, device=args.lam_dev, id='lamina',
              debug=False)

# check core.py on how to connect 2 modules
# amacrine cells get input 
# from photoreceptors and give output to other neurons
# see synapse_lamina.csv

man.add_mod(lpu_ret)
man.add_mod(lpu_lam)
print('Starting simulation')
man.start(steps=1000)
print('Simulation complete')
man.stop()