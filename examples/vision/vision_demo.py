#!/usr/bin/env python

"""
Vision system demo.

Notes
-----
Generate input file by running ./data/gen_vis_input.py
and generate configurations of LPUs by running ./data/generate_vision_gexf.py
"""

import argparse
import itertools

import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU
import data.vision_configuration as vc

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
parser.add_argument('-r', '--time_sync', default=False, action='store_true',
                    help='Time data reception throughput [default: False]')
parser.add_argument('-a', '--lam_dev', default=0, type=int,
                    help='GPU for lamina lobe [default: 0]')
parser.add_argument('-m', '--med_dev', default=1, type=int,
                    help='GPU for medulla [default: 1]')

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
man.add(LPU, 'lamina', dt, n_dict_lam, s_dict_lam,
        input_file='./data/vision_input.h5',
        output_file='lamina_output.h5', 
        device=args.lam_dev, time_sync=args.time_sync)

(n_dict_med, s_dict_med) = LPU.lpu_parser('./data/medulla.gexf.gz')
man.add(LPU, 'medulla', dt, n_dict_med, s_dict_med,
        output_file='medulla_output.h5', 
        device=args.med_dev, time_sync=args.time_sync)

pat = vc.create_pattern(n_dict_lam, n_dict_med)

man.connect('lamina', 'medulla', pat, 0, 1, compat_check=False)

man.spawn()
man.start(steps=args.steps)
man.wait()
