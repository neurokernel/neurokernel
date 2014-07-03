#!/usr/bin/env python

"""
Generic LPU demo

Notes
-----
Generate input file and LPU configuration by running

cd data
python gen_generic_lpu.py
"""

import argparse
import itertools

import networkx as nx

import neurokernel.core as core
import neurokernel.base as base
from neurokernel.tools.comm import get_random_port

from neurokernel.LPU.LPU import LPU

dt = 1e-4
dur = 1.0
steps = int(dur/dt)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=True,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=steps, type=int,
                    help='Number of steps [default: %s]' % steps)
parser.add_argument('-d', '--port_data', default=None, type=int,
                    help='Data port [default: randomly selected]')
parser.add_argument('-c', '--port_ctrl', default=None, type=int,
                    help='Control port [default: randomly selected]')
parser.add_argument('-g', '--gpu_dev', default=0, type=int,
                    help='GPU device number [default: 0]')
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = base.setup_logger(file_name, screen)

if args.port_data is None and args.port_ctrl is None:
    port_data = get_random_port()
    port_ctrl = get_random_port()
else:
    port_data = args.port_data
    port_ctrl = args.port_ctrl

man = core.Manager(port_data, port_ctrl)
man.add_brok()

(n_dict, s_dict) = LPU.lpu_parser('./data/generic_lpu.gexf.gz')

ge = LPU(dt, n_dict, s_dict,
         input_file='./data/generic_input.h5',
         output_file='generic_output.h5', port_ctrl=port_ctrl,
         port_data=port_data,
         device=args.gpu_dev, id='ge',
         debug=args.debug)
man.add_mod(ge)

man.start(steps=args.steps)
man.stop()
