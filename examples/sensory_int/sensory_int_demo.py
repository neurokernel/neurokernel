#!/usr/bin/env python

"""
Sensory integration model demo.

Notes
-----
Set integration module neuron parameters by running ./data/create_int_gexf.py
Generate input files by running ./data/gen_olf_input.py and ./data/gen_vis_input.py
"""

import argparse
import itertools

import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

import neurokernel.pattern as pattern
import neurokernel.plsel as plsel
from neurokernel.LPU.LPU import LPU
import data.vision_configuration as vc

import neurokernel.mpi_relaunch

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=14000, type=int,
                    help='Number of steps [default:14000]')
parser.add_argument('-r', '--time_sync', default=False, action='store_true',
                    help='Time data reception throughput [default: False]')
parser.add_argument('-b', '--lam_dev', default=0, type=int,
                    help='GPU for lamina lobe [default:0]')
parser.add_argument('-m', '--med_dev', default=1, type=int,
                    help='GPU for medulla [default:1]')
parser.add_argument('-a', '--al_dev', default=2, type=int,
                    help='GPU for antennal lobe [default:2]')
parser.add_argument('-i', '--int_dev', default=3, type=int,
                    help='GPU for integration [default:3]')
args = parser.parse_args()

dt = 1e-4
dur = 1.4
Nt = args.steps or int(dur/dt)

file_name = 'neurokernel.log' if args.log.lower() in ['file', 'both'] else None
screen = True if args.log.lower() in ['screen', 'both'] else False
logger = setup_logger(file_name=file_name, screen=screen)

man = core.Manager()

# Load configurations for lamina, medulla and antennal lobe models:
al_id = 'antennallobe'
(n_dict_al, s_dict_al) = LPU.lpu_parser( './data/antennallobe.gexf.gz')
man.add(LPU, al_id, dt, n_dict_al, s_dict_al,
        input_file='./data/olfactory_input.h5',
        output_file='antennallobe_output.h5',
        device=args.al_dev, time_sync=args.time_sync)

lam_id = 'lamina'
(n_dict_lam, s_dict_lam) = LPU.lpu_parser('./data/lamina.gexf.gz')
man.add(LPU, lam_id, dt, n_dict_lam, s_dict_lam,
        input_file='./data/vision_input.h5',
        output_file='lamina_output.h5',
        device=args.al_dev, time_sync=args.time_sync)

med_id = 'medulla'
(n_dict_med, s_dict_med) = LPU.lpu_parser('./data/medulla.gexf.gz')
man.add(LPU, med_id, dt, n_dict_med, s_dict_med,
        output_file='medulla_output.h5',
        device=args.al_dev, time_sync=args.time_sync)

int_id = 'integrate'
(n_dict_int, s_dict_int) = LPU.lpu_parser('./data/integrate.gexf.gz')
man.add(LPU, int_id, dt, n_dict_int, s_dict_int,
        output_file='integrate_output.h5',
        device=args.al_dev, time_sync=args.time_sync)

# Connect lamina to medulla:
pat_lam_med = vc.create_pattern(n_dict_lam, n_dict_med)
man.connect(lam_id, med_id, pat_lam_med, 0, 1, compat_check=False)

# Create connectivity patterns between antennal lobe, medulla, and 
# integration LPUs:
sel_al = LPU.extract_all(n_dict_al)
sel_int = LPU.extract_all(n_dict_int)
sel_med = LPU.extract_all(n_dict_med)

pat_al_int = pattern.Pattern(sel_al, sel_int)
pat_med_int = pattern.Pattern(sel_med, sel_int)

# Define connections from antennal lobe to integration LPU:
for src, dest in zip(['/al[0]/pn%d' % i for i in xrange(3)],
                     plsel.Selector(LPU.extract_in_spk(n_dict_int))):
    pat_al_int[src, dest] = 1
    pat_al_int.interface[src, 'type'] = 'spike'
    pat_al_int.interface[dest, 'type'] = 'spike'

# Define connections from medulla to integration LPU:
for src, dest in zip(['/medulla/Mt3%c[%d]' % (c, i) \
                      for c in ('h','v') for i in xrange(4)],
                     plsel.Selector(LPU.extract_in_gpot(n_dict_int))):
    pat_med_int[src, dest] = 1
    pat_med_int.interface[src, 'type'] = 'gpot'
    pat_med_int.interface[dest, 'type'] = 'gpot'

man.connect(al_id, int_id, pat_al_int, 0, 1, compat_check=False)
man.connect(med_id, int_id, pat_med_int, 0, 1, compat_check=False)

man.spawn()
man.start(steps=Nt)
man.wait()
