#!/usr/bin/env python

"""
Early olfactory system demo.

Notes
-----
Generate input file by running ./data/gen_olf_input.py
"""

import argparse
import itertools

import networkx as nx

import neurokernel.core as core
import neurokernel.base as base
import neurokernel.tools.graph as graph_tools

from neurokernel.LPU.lpu_parser import lpu_parser
from neurokernel.LPU.LPU_rev import LPU_rev

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file,screen,both,or none;default:none]')
parser.add_argument('-s', '--steps', default=10000, type=int,
                    help='Number of steps [default:10000]')
parser.add_argument('-d', '--data_port', default=5005, type=int,
                    help='Data port [default:5005]')
parser.add_argument('-c', '--ctrl_port', default=5006, type=int,
                    help='Control port [default:5006]')
parser.add_argument('-a', '--al_dev', default=0, type=int,
                    help='GPU for antennal lobe [default:0]')
args = parser.parse_args()

dt = 1e-4
dur = 1.0
Nt = int(dur/dt)

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True    
logger = base.setup_logger(file_name, screen)

man = core.Manager(port_data=args.data_port, port_ctrl=args.ctrl_port)
man.add_brok()

(n_dict, s_dict) = LPU_rev.lpu_parser('./data/antennallobe.gexf.gz')

al = LPU_rev(dt, n_dict, s_dict, input_file='./data/olfactory_input.h5',
             output_file='olfactory_output.h5', port_ctrl=man.port_ctrl,
             port_data=man.port_data,
             device=args.al_dev, id='al',
             debug=args.debug)
man.add_mod(al)

man.start(steps=args.steps)
man.stop()
