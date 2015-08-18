#!/usr/bin/env python

"""
Generate and run generic LPU on multiple GPUs.
"""

import argparse
import itertools
import random

import numpy as np

import data.gen_generic_lpu as g

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

import neurokernel.pattern as pattern
import neurokernel.plsel as plsel
from neurokernel.LPU.LPU import LPU

import neurokernel.mpi_relaunch

# Execution parameters:
dt = 1e-4
dur = 1.0
start = 0.3
stop = 0.6
I_max = 0.6
steps = int(dur/dt)

N_sensory = 30
N_local = 30
N_output = 30

num_lpus = 2

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
parser.add_argument('-y', '--num_sensory', default=N_sensory, type=int,
                    help='Number of sensory neurons associated with LPU 0 [default: %s]' % N_sensory)
parser.add_argument('-n', '--num_local', default=N_local, type=int,
                    help='Number of local neurons in each LPU [default: %s]' % N_local)
parser.add_argument('-o', '--num_output', default=N_output, type=int,
                    help='Number of output neurons in each LPU [default: %s]' % N_output)
parser.add_argument('-u', '--num_lpus', default=num_lpus, type=int,
                    help='Number of LPUs [default: %s]' % num_lpus)
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name=file_name, screen=screen)

# Set number of local and projection neurons in each LPU:
N = args.num_lpus
neu_dict = {i: [0, args.num_local, args.num_output] for i in xrange(N)}

# Only LPU 0 receives input and should therefore be associated with a population
# of sensory neurons:
neu_dict[0][0] = args.num_sensory

# Initialize RNG:
random.seed(0)
np.random.seed(0)

# Create input signal for LPU 0:
in_file_name_0 = 'generic_input.h5'
g.create_input(in_file_name_0, neu_dict[0][0], dt, dur, start, stop, I_max)

# Store info for all instantiated LPUs in the following dict:
lpu_dict = {}

# Set up several LPUs:
man = core.Manager()
for i, neu_num in neu_dict.iteritems():
    lpu_entry = {}

    if i == 0:
        in_file_name = in_file_name_0
    else:
        in_file_name = None
    lpu_file_name = 'generic_lpu_%s.gexf.gz' % i
    out_file_name = 'generic_lpu_%s_output.h5' % i

    id = 'lpu_%s' % i

    g.create_lpu(lpu_file_name, id, *neu_num)
    (n_dict, s_dict) = LPU.lpu_parser(lpu_file_name)

    man.add(LPU, id, dt, n_dict, s_dict,
            input_file=in_file_name,
            output_file=out_file_name,
            device=i,
            debug=args.debug)

    lpu_entry['lpu_file_name'] = lpu_file_name
    lpu_entry['in_file_name'] = in_file_name
    lpu_entry['out_file_name'] = out_file_name
    lpu_entry['n_dict'] = n_dict
    lpu_entry['s_dict'] = s_dict

    lpu_dict[id] = lpu_entry

# Create connectivity patterns between each combination of LPU pairs:
for id_0, id_1 in itertools.combinations(lpu_dict.keys(), 2):

    n_dict_0 = lpu_dict[id_0]['n_dict']
    n_dict_1 = lpu_dict[id_1]['n_dict']

    # Find all output and input port selectors in each LPU:
    out_ports_spk_0 = plsel.Selector(LPU.extract_out_spk(n_dict_0))
    out_ports_gpot_0 = plsel.Selector(LPU.extract_out_gpot(n_dict_0))

    out_ports_spk_1 = plsel.Selector(LPU.extract_out_spk(n_dict_1))
    out_ports_gpot_1 = plsel.Selector(LPU.extract_out_gpot(n_dict_1))

    in_ports_spk_0 = plsel.Selector(LPU.extract_in_spk(n_dict_0))
    in_ports_gpot_0 = plsel.Selector(LPU.extract_in_gpot(n_dict_0))

    in_ports_spk_1 = plsel.Selector(LPU.extract_in_spk(n_dict_1))
    in_ports_gpot_1 = plsel.Selector(LPU.extract_in_gpot(n_dict_1)) 

    out_ports_0 = plsel.Selector.union(out_ports_spk_0, out_ports_gpot_0)
    out_ports_1 = plsel.Selector.union(out_ports_spk_1, out_ports_gpot_1)

    in_ports_0 = plsel.Selector.union(in_ports_spk_0, in_ports_gpot_0)
    in_ports_1 = plsel.Selector.union(in_ports_spk_1, in_ports_gpot_1)

    # Initialize a connectivity pattern between the two sets of port
    # selectors:
    pat = pattern.Pattern(plsel.Selector.union(out_ports_0, in_ports_0),
                          plsel.Selector.union(out_ports_1, in_ports_1))

    # Create connections from the ports with identifiers matching the output
    # ports of one LPU to the ports with identifiers matching the input
    # ports of the other LPU. First, define connections from LPU0 to LPU1:
    N_conn_spk_0_1 = min(len(out_ports_spk_0), len(in_ports_spk_1))
    N_conn_gpot_0_1 = min(len(out_ports_gpot_0), len(in_ports_gpot_1))
    for src, dest in zip(random.sample(out_ports_spk_0.identifiers,
                                       N_conn_spk_0_1), 
                         random.sample(in_ports_spk_1.identifiers,
                                       N_conn_spk_0_1)):
        pat[src, dest] = 1
        pat.interface[src, 'type'] = 'spike'
        pat.interface[dest, 'type'] = 'spike'
    for src, dest in zip(random.sample(out_ports_gpot_0.identifiers,
                                       N_conn_gpot_0_1),
                         random.sample(in_ports_gpot_1.identifiers,
                                       N_conn_gpot_0_1)):
        pat[src, dest] = 1
        pat.interface[src, 'type'] = 'gpot'
        pat.interface[dest, 'type'] = 'gpot'

    # Next, define connections from LPU1 to LPU0:
    N_conn_spk_1_0 = min(len(out_ports_spk_1), len(in_ports_spk_0))
    N_conn_gpot_1_0 = min(len(out_ports_gpot_1), len(in_ports_gpot_0))
    for src, dest in zip(random.sample(out_ports_spk_1, N_conn_spk_1_0), 
                         random.sample(in_ports_spk_0, N_conn_spk_1_0)):
        pat[src, dest] = 1
        pat.interface[src, 'type'] = 'spike'
        pat.interface[dest, 'type'] = 'spike'
    for src, dest in zip(random.sample(out_ports_gpot_1, N_conn_gpot_1_0),
                         random.sample(in_ports_gpot_0, N_conn_gpot_1_0)):
        pat[src, dest] = 1
        pat.interface[src, 'type'] = 'gpot'
        pat.interface[dest, 'type'] = 'gpot'

    man.connect(id_0, id_1, pat, 0, 1, compat_check=False)

man.spawn()
man.start(steps=steps)
man.wait()
