#!/usr/bin/env python

"""
Introductory Neurokernel demo

Notes
-----
Generate input files and LPU configurations by running

cd data
python gen_generic_lpu.py -s 0 -l lpu_0 generic_lpu_0.gexf.gz generic_lpu_0_input.h5
python gen_generic_lpu.py -s 1 -l lpu_1 generic_lpu_1.gexf.gz generic_lpu_1_input.h5

Other seed values may be specified, but note that some may result in a network
that generates no meaningful responses to the input signal.
"""

import argparse
import futures
import itertools
import random

import networkx as nx

import neurokernel.core as core
import neurokernel.base as base
from neurokernel.tools.comm import get_random_port

import neurokernel.pattern as pattern
from neurokernel.LPU.LPU import LPU

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
parser.add_argument('-t', '--port_time', default=None, type=int,
                    help='Timing port [default: randomly selected]')
parser.add_argument('-r', '--time_sync', default=False, action='store_true',
                    help='Time data reception throughput [default: False]')
parser.add_argument('-g', '--gpu_dev', default=[0, 1], type=int, nargs='+',
                    help='GPU device numbers [default: 0 1]')
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = base.setup_logger(file_name, screen)

def run(connected):
    if args.port_data is None:        
        port_data = get_random_port()
    else:
        port_data = args.port_data
    if args.port_ctrl is None:
        port_ctrl = get_random_port()
    else:
        port_ctrl = args.port_ctrl
    if args.port_time is None:
        port_time = get_random_port()
    else:
        port_time = args.port_time

    out_name = 'un' if not connected else 'co'
    man = core.Manager(port_data, port_ctrl, port_time)
    man.add_brok()

    lpu_file_0 = './data/generic_lpu_0.gexf.gz'
    lpu_file_1 = './data/generic_lpu_1.gexf.gz'
    (n_dict_0, s_dict_0) = LPU.lpu_parser(lpu_file_0)
    (n_dict_1, s_dict_1) = LPU.lpu_parser(lpu_file_1)

    lpu_0_id = 'lpu_0'
    lpu_0 = LPU(dt, n_dict_0, s_dict_0,
                input_file='./data/generic_lpu_0_input.h5',
                output_file='generic_lpu_0_%s_output.h5' % out_name,
                port_ctrl=port_ctrl, port_data=port_data,
                port_time=port_time,
                device=args.gpu_dev[0], id=lpu_0_id,
                debug=args.debug, time_sync=args.time_sync)
    man.add_mod(lpu_0)

    lpu_1_id = 'lpu_1'
    lpu_1 = LPU(dt, n_dict_1, s_dict_1,
                input_file='./data/generic_lpu_1_input.h5',
                output_file='generic_lpu_1_%s_output.h5' % out_name,
                port_ctrl=port_ctrl, port_data=port_data,
                port_time=port_time,
                device=args.gpu_dev[1], id=lpu_1_id,
                debug=args.debug, time_sync=args.time_sync)
    man.add_mod(lpu_1)

    # Create random connections between the input and output ports if the LPUs
    # are to be connected:
    if connected:

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
        # ports of the other LPU:
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

        man.connect(lpu_0, lpu_1, pat, 0, 1)

    man.start(steps=args.steps)
    man.stop()

random.seed(0)
with futures.ProcessPoolExecutor() as executor:
    for connected in [False, True]:
        executor.submit(run, connected)

