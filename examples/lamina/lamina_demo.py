#!/usr/bin/env python

"""
Vision system demo.

Notes
-----
Generate input file by running ./data/gen_vis_input.py
"""

import argparse
import itertools

import networkx as nx

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU

import neurokernel.mpi_relaunch

dt = 1e-4
dur = 1.0
steps = int(dur/dt)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
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
parser.add_argument('-a', '--lam_dev', default=0, type=int,
                    help='GPU for lamina lobe [default: 0]')

args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name=file_name, screen=screen)

man = core.Manager()

(n_dict_lam, s_dict_lam) = LPU.lpu_parser('./data/lamina.gexf.gz')
man.add(LPU, 'lpu_lam', dt, n_dict_lam, s_dict_lam,
        input_file='./data/vision_input.h5',
        output_file='lamina_output.h5', 
        debug=args.debug, device=args.lam_dev)

man.spawn()
man.start(steps=args.steps)
man.wait()
