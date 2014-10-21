#!/usr/bin/env python

"""
Generate and run generic LPU on multiple GPUs.
"""

import argparse
import itertools
import random

import networkx as nx

import data.gen_generic_lpu as g

import neurokernel.core as core
import neurokernel.base as base
from neurokernel.tools.comm import get_random_port

import neurokernel.pattern as pattern
from neurokernel.LPU.LPU import LPU

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
parser.add_argument('-d', '--port_data', default=None, type=int,
                    help='Data port [default: randomly selected]')
parser.add_argument('-c', '--port_ctrl', default=None, type=int,
                    help='Control port [default: randomly selected]')
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
logger = base.setup_logger(file_name, screen)

# Set number of local and projection neurons in each LPU:
N = args.num_lpus
neu_dict = {i: [0, args.num_local, args.num_output] for i in xrange(N)}

# Only LPU 0 receives input and should therefore be associated with a population
# of sensory neurons:
neu_dict[0][0] = args.num_sensory

# Create input signal for LPU 0:
in_file_name_0 = 'generic_input.h5'
g.create_input(in_file_name_0, neu_dict[0][0], dt, dur, start, stop, I_max)

# Store info for all instantiated LPUs in the following dict:
lpu_dict = {}

# Create several LPUs:
if args.port_data is None and args.port_ctrl is None:
    port_data = get_random_port()
    port_ctrl = get_random_port()
else:
    port_data = args.port_data
    port_ctrl = args.port_ctrl

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

    lpu = LPU(dt, n_dict, s_dict, input_file=in_file_name,
              output_file=out_file_name,
              port_ctrl=port_ctrl, port_data=port_data,
              device=i, id=id,
              debug=args.debug)

    lpu_entry['lpu_file_name'] = lpu_file_name
    lpu_entry['in_file_name'] = in_file_name
    lpu_entry['out_file_name'] = out_file_name
    lpu_entry['lpu'] = lpu
    lpu_entry['id'] = id

    lpu_dict[i] = lpu_entry

man = core.Manager(port_data, port_ctrl)
man.add_brok()

random.seed(0)

# Since each connectivity pattern between two LPUs contains the synapses in both
# directions, create connectivity patterns between each combination of LPU
# pairs:
for id_0, id_1 in itertools.combinations(lpu_dict.keys(), 2):

    lpu_0 = lpu_dict[id_0]['lpu']
    lpu_1 = lpu_dict[id_1]['lpu']

    # Find all output and input port selectors in each LPU:
    out_ports_0 = lpu_0.interface.out_ports().to_selectors()
    out_ports_1 = lpu_1.interface.out_ports().to_selectors()

    in_ports_0 = lpu_0.interface.in_ports().to_selectors()
    in_ports_1 = lpu_1.interface.in_ports().to_selectors()

    out_ports_spk_0 = lpu_0.interface.out_ports().spike_ports().to_selectors()
    out_ports_gpot_0 = lpu_0.interface.out_ports().gpot_ports().to_selectors()

    out_ports_spk_1 = lpu_1.interface.out_ports().spike_ports().to_selectors()
    out_ports_gpot_1 = lpu_1.interface.out_ports().gpot_ports().to_selectors()

    in_ports_spk_0 = lpu_0.interface.in_ports().spike_ports().to_selectors()
    in_ports_gpot_0 = lpu_0.interface.in_ports().gpot_ports().to_selectors()

    in_ports_spk_1 = lpu_1.interface.in_ports().spike_ports().to_selectors()
    in_ports_gpot_1 = lpu_1.interface.in_ports().gpot_ports().to_selectors()

    # Initialize a connectivity pattern between the two sets of port
    # selectors:
    pat = pattern.Pattern(','.join(out_ports_0+in_ports_0),
                          ','.join(out_ports_1+in_ports_1))

    # Create connections from the ports with identifiers matching the output
    # ports of one LPU to the ports with identifiers matching the input
    # ports of the other LPU. First, define connections from LPU0 to LPU1:
    N_conn_spk_0_1 = min(len(out_ports_spk_0), len(in_ports_spk_1))
    N_conn_gpot_0_1 = min(len(out_ports_gpot_0), len(in_ports_gpot_1))
    for src, dest in zip(random.sample(out_ports_spk_0, N_conn_spk_0_1), 
                         random.sample(in_ports_spk_1, N_conn_spk_0_1)):
        pat[src, dest] = 1
        pat.interface[src, 'type'] = 'spike'
        pat.interface[dest, 'type'] = 'spike'
    for src, dest in zip(random.sample(out_ports_gpot_0, N_conn_gpot_0_1),
                         random.sample(in_ports_gpot_1, N_conn_gpot_0_1)):
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

    man.connect(lpu_0, lpu_1, pat, 0, 1)

man.start(steps=steps)
man.stop()
