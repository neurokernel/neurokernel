#!/usr/bin/env python

import argparse

import neurokernel.core as core
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.comm import get_random_port

from data.gen_vis_input import generate_input
from data.gen_vis_gexf import generate_gexf



parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rec-micro', dest='record_microvilli',
                    action="store_true", help='records microvilli if set')
parser.add_argument('-u', '--unrec-neuro', dest='record_neuron',
                    action="store_false",
                    help='does not record neuron if set')

parser.add_argument('-l', '--layers', dest='num_layers', type=int,
                    default=17,
                    help='number of layers of ommatidia on circle')
parser.add_argument('-m', '--micro', dest='num_microvilli', type=int,
                    default=30000,
                    help='number of microvilli in each photoreceptor')

parser.add_argument('-d', '--port_data', default=None, type=int,
                    help='Data port [default: randomly selected]')
parser.add_argument('-c', '--port_ctrl', default=None, type=int,
                    help='Control port [default: randomly selected]')
parser.add_argument('-a', '--ret_dev', default=0, type=int,
                    help='GPU for lamina lobe [default: 0]')

parser.add_argument('-i', '--input', action="store_true",
                    help='generates input if set')
parser.add_argument('-g', '--gexf', action="store_true",
                    help='generates gexf of LPU if set')

args = parser.parse_args()

dt = 1e-4
GEXF_FILE = 'retina.gexf.gz'
INPUT_FILE = 'vision_input.h5'
IMAGE_FILE = 'image1.mat'

if args.input:
    generate_input(INPUT_FILE, IMAGE_FILE, args.num_layers)
if args.gexf:
    n = args.num_layers
    photoreceptor_num = 3*n*(n+1)+1
    generate_gexf(GEXF_FILE, photoreceptor_num)

if args.port_data is None and args.port_ctrl is None:
    port_data = get_random_port()
    port_ctrl = get_random_port()
else:
    port_data = args.port_data
    port_ctrl = args.port_ctrl

man = core.Manager(port_data, port_ctrl)
man.add_brok()

(n_dict_ret, s_dict_ret) = LPU.lpu_parser(GEXF_FILE)
lpu_ret = LPU(dt, n_dict_ret, s_dict_ret,
              input_file=INPUT_FILE,
              output_file='retina_output.h5', port_ctrl=port_ctrl,
              port_data=port_data, device=args.ret_dev, id='retina',
              debug=False)

man.add_mod(lpu_ret)

man.start(steps=1000)
man.stop()