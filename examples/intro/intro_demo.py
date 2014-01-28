#!/usr/bin/env python

"""
Introductory Neurokernel demo

Notes
-----
Generate input files and LPU configurations by running

cd data
python gen_generic_lpu.py -s 0 generic_lpu_0.gexf.gz generic_input_0.h5
python gen_generic_lpu.py -s 1 generic_lpu_1.gexf.gz generic_input_1.h5
"""

import argparse
import futures
import itertools

import networkx as nx

import neurokernel.core as core
import neurokernel.base as base
from neurokernel.tools.comm import get_random_port

from neurokernel.LPU.lpu_parser import lpu_parser
from neurokernel.LPU.LPU_rev import LPU_rev

import neurokernel.tools.graph

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=10000, type=int,
                    help='Number of steps [default:10000]')
parser.add_argument('-d', '--port_data', default=None, type=int,
                    help='Data port [default:randomly selected]')
parser.add_argument('-c', '--port_ctrl', default=None, type=int,
                    help='Control port [default:randomly selected]')
parser.add_argument('-g', '--gpu_dev', default=[0, 1], type=int, nargs='+',
                    help='GPU device numbers [default:[0, 1]]')
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

def run(connected):
    if args.port_data is None and args.port_ctrl is None:
        port_data = get_random_port()
        port_ctrl = get_random_port()
    else:
        port_data = args.port_data
        port_ctrl = args.port_ctrl

    out_name = 'un' if not connected else 'co'
    man = core.Manager(port_data, port_ctrl)
    man.add_brok()

    lpu_file_0 = './data/generic_lpu_0.gexf.gz'
    lpu_file_1 = './data/generic_lpu_1.gexf.gz'
    (n_dict_0, s_dict_0) = LPU_rev.lpu_parser(lpu_file_0)
    (n_dict_1, s_dict_1) = LPU_rev.lpu_parser(lpu_file_1)

    ge_0_id = 'ge_0'
    ge_0 = LPU_rev(dt, n_dict_0, s_dict_0,
                   input_file='./data/generic_input_0.h5',
                   output_file='generic_output_0_%s.h5' % out_name,
                   port_ctrl=port_ctrl, port_data=port_data,
                   device=args.gpu_dev[0], id=ge_0_id,
                   debug=args.debug)
    man.add_mod(ge_0)

    ge_1_id = 'ge_1'
    ge_1 = LPU_rev(dt, n_dict_1, s_dict_1,
                   input_file='./data/generic_input_1.h5',
                   output_file='generic_output_1_%s.h5' % out_name,
                   port_ctrl=port_ctrl, port_data=port_data,
                   device=args.gpu_dev[1], id=ge_1_id,
                   debug=args.debug)
    man.add_mod(ge_1)

    # Connect the public neurons in the two LPUs:
    df_neu_0, df_syn_0 = neurokernel.tools.graph.graph_to_df(nx.read_gexf(lpu_file_0))
    df_neu_1, df_syn_1 = neurokernel.tools.graph.graph_to_df(nx.read_gexf(lpu_file_1))

    # Number of public neurons in each LPU:
    N_spike_0 = len(df_neu_0[(df_neu_0['spiking']==True)&(df_neu_0['public']==True)])
    N_gpot_0 = len(df_neu_0[(df_neu_0['spiking']==False)&(df_neu_0['public']==True)])

    N_spike_1 = len(df_neu_1[(df_neu_1['spiking']==True)&(df_neu_1['public']==True)])
    N_gpot_1 = len(df_neu_1[(df_neu_1['spiking']==False)&(df_neu_1['public']==True)])

    # Alpha function synaptic parameters:
    alphasynapse_type_params = {'AlphaSynapse': ['ad', 'ar', 'gmax', 'id', 'class', 'conductance',
                                                 'reverse']}

    if connected:
        conn = core.Connectivity(N_gpot_0, N_spike_0, N_gpot_1, N_spike_1, 1,
                                 ge_0.id, ge_1.id, alphasynapse_type_params)
        for id, (i, j) in enumerate(itertools.product(xrange(N_spike_0), xrange(N_spike_1))):
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j] = 1
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'name'] = 'int_0to1_%s_%s' % (i, j)
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'model'] = 'AlphaSynapse'

            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'ad'] = 0.19*1000
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'ar'] = 1.1*100
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'class'] = 0
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'conductance'] = True
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'gmax'] = 0.003
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'id'] = id
            conn[ge_0_id, 'spike', i, ge_1_id, 'spike', j, 0, 'reverse'] = 0.065

        man.connect(ge_0, ge_1, conn)

    man.start(steps=args.steps)
    man.stop()

with futures.ProcessPoolExecutor() as executor:
    for connected in [False, True]:
        executor.submit(run, connected)
