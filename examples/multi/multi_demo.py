#!/usr/bin/env python

"""
Generate and run generic LPU on multiple GPUs.
"""

import argparse
import itertools
import time

import numpy as np
import networkx as nx

import gen_generic_lpu as g

from neurokernel.tools.graph import graph_to_df
from neurokernel.tools.comm import get_random_port
from neurokernel.base import setup_logger
from neurokernel.core import Connectivity, Manager
from neurokernel.LPU.LPU_rev import LPU_rev

# Execution parameters:
dt = 1e-4
dur = 1.0
start = 0.3
stop = 0.6
I_max = 0.6
steps = int(dur/dt)

num_sensory = 30
num_local = 30
num_output = 30

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
parser.add_argument('-y', '--num_sensory', default=num_sensory, type=int,
                    help='Number of sensory neurons associated with LPU 0 [default: %s]' % num_sensory)
parser.add_argument('-n', '--num_local', default=num_local, type=int,
                    help='Number of local neurons in each LPU [default: %s]' % num_local)
parser.add_argument('-o', '--num_output', default=num_output, type=int,
                    help='Number of output neurons in each LPU [default: %s]' % num_output)
parser.add_argument('-u', '--num_lpus', default=num_lpus, type=int,
                    help='Number of LPUs [default: %s]' % num_lpus)
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name, screen)

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
    out_file_name = 'generic_output_%s.h5' % i
    
    g.create_lpu(lpu_file_name, *neu_num)
    (n_dict, s_dict) = LPU_rev.lpu_parser(lpu_file_name)

    id = 'lpu_%s' % i
    lpu = LPU_rev(dt, n_dict, s_dict, input_file=in_file_name,
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
    
syn_params = {'AlphaSynapse': ['ad', 'ar', 'gmax', 'id', 'class', 'conductance', 'reverse']}

man = Manager(port_data, port_ctrl)
man.add_brok()

# Since each connectivity pattern between two LPUs contains the synapses in both
# directions, create connectivity patterns between each combination of LPU
# pairs:
for lpu_0, lpu_1 in itertools.combinations(lpu_dict.keys(), 2):

    df_neu_0, df_syn_0 = graph_to_df(nx.read_gexf(lpu_dict[lpu_0]['lpu_file_name']))
    df_neu_1, df_syn_1 = graph_to_df(nx.read_gexf(lpu_dict[lpu_1]['lpu_file_name']))

    N_spike_0 = len(df_neu_0[(df_neu_0['spiking']==True)&(df_neu_0['public']==True)])
    N_gpot_0 = len(df_neu_0[(df_neu_0['spiking']==False)&(df_neu_0['public']==True)])

    N_spike_1 = len(df_neu_1[(df_neu_1['spiking']==True)&(df_neu_1['public']==True)])
    N_gpot_1 = len(df_neu_1[(df_neu_1['spiking']==False)&(df_neu_1['public']==True)])

    conn = Connectivity(N_gpot_0, N_spike_0, N_gpot_1, N_spike_1, 1,
                        lpu_dict[lpu_0]['id'], lpu_dict[lpu_1]['id'],
                        syn_params)

    # Define synapses between spiking neurons in both directions:
    for id_src, id_dest, N_spike_src, N_spike_dest in \
      [(lpu_dict[lpu_0]['id'], lpu_dict[lpu_1]['id'], N_spike_0, N_spike_1),
        (lpu_dict[lpu_1]['id'], lpu_dict[lpu_0]['id'], N_spike_1, N_spike_0)]:
        id_start = 0
        for id, (i, j) in enumerate(itertools.product(xrange(N_spike_src),
                                                      xrange(N_spike_dest)), id_start):
            conn[id_src, 'spike', i, id_dest, 'spike', j] = 1
            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'name'] = \
                'syn_%s:%s_%s:%s' % (id_src, i, id_dest, j)
            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'model'] = 'AlphaSynapse'

            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'ad'] = 0.19*1000
            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'ar'] = 1.1*100
            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'class'] = 0 # spike->spike            
            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'conductance'] = True
            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'gmax'] = 0.003
            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'id'] = id
            conn[id_src, 'spike', i, id_dest, 'spike', j, 0, 'reverse'] = 0.065
        id_start = id+1
        
    man.connect(lpu_dict[lpu_0]['lpu'], lpu_dict[lpu_1]['lpu'], conn)

start = time.time()
man.start(steps=steps)        
man.stop()
print 'time: ', time.time()-start
