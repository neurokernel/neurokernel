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

import neurokernel.core as core
import neurokernel.base as base
import neurokernel.tools.graph as graph_tools

from neurokernel.LPU.lpu_parser import lpu_parser
from neurokernel.LPU.LPU_rev import LPU_rev
from neurokernel.LPU.LPU import LPU

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=10000, type=int,
                    help='Number of steps [default:10000]')
parser.add_argument('-d', '--data_port', default=5005, type=int,
                    help='Data port [default:5005]')
parser.add_argument('-c', '--ctrl_port', default=5006, type=int,
                    help='Control port [default:5006]')
parser.add_argument('-b', '--lam_dev', default=0, type=int,
                    help='GPU for lamina lobe [default:0]')
parser.add_argument('-a', '--al_dev', default=1, type=int,
                    help='GPU for antennal lobe [default:1]')
parser.add_argument('-m', '--med_dev', default=2, type=int,
                    help='GPU for medulla [default:2]')
parser.add_argument('-i', '--int_dev', default=3, type=int,
                    help='GPU for integration [default:3]')

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

(n_dict_al, s_dict_al) = LPU_rev.lpu_parser( './data/antennallobe.gexf.gz')
lpu_al = LPU_rev(dt, n_dict_al, s_dict_al,
                 input_file='./data/olfactory_input.h5',
                 output_file='antennallobe_output.h5', port_ctrl=man.port_ctrl,
                 port_data=man.port_data, device=args.al_dev, id='antennallobe')
man.add_mod(lpu_al)

(n_dict_lam, s_dict_lam) = lpu_parser('./data/lamina.gexf.gz')
lpu_lam = LPU(dt, n_dict_lam, s_dict_lam,
              input_file='./data/vision_input.h5',
              output_file='lamina_output.h5', port_ctrl= man.port_ctrl,
              port_data=man.port_data, device=args.lam_dev, id='lamina')
man.add_mod(lpu_lam)

(n_dict_med, s_dict_med) = lpu_parser('./data/medulla.gexf.gz')
lpu_med = LPU(dt, n_dict_med, s_dict_med,
              output_file='medulla_output.h5', port_ctrl= man.port_ctrl,
              port_data=man.port_data, device=args.med_dev, id='medulla')
man.add_mod(lpu_med)

(n_dict_int, s_dict_int) = LPU_rev.lpu_parser('./data/integrate.gexf.gz')
lpu_int = LPU_rev(dt, n_dict_int, s_dict_int,
                  output_file='integrate_output.h5', port_ctrl= man.port_ctrl,
                  port_data=man.port_data, device=args.int_dev, id='integrate')

g = nx.read_gexf('./data/lamina_medulla.gexf.gz', relabel=True)
conn_lam_med = graph_tools.graph_to_conn(g)
man.connect(lpu_lam, lpu_med, conn_lam_med)

N_med_gpot = 3072  # number of public graded potential medulla neurons
N_med_spike = 16   # number of public spiking lobula neurons
N_al = 1540        # number of public antennal lobe neurons
N_al_pn = 165      # number of antennal lobe projection neurons
N_int = 10         # number of public integration neurons

# Create unidirectional connection patterns between
# vision -> integration and olfaction -> integration:
med_id = 'medulla'
al_id = 'antennallobe'
int_id = 'integrate'

alphasynapse_type_params = {'AlphaSynapse': ['reverse', 'gmax', 'conductance',
                                             'id', 'ar', 'ad', 'class']}
conn_med_int = core.Connectivity(N_med_gpot, N_med_spike,
                                 0, N_int, 1,
                                 med_id, int_id, alphasynapse_type_params)
sl_med = slice(0, N_med_spike)
sl_int = slice(0, N_int)
for id, (i, j) in enumerate(itertools.product(xrange(N_med_spike),
                                              xrange(N_int))):
    conn_med_int[med_id, 'spike', i, int_id, 'spike', j] = 1
    conn_med_int[med_id, 'spike', i, int_id, 'spike', j, 0, 'id'] = id
    conn_med_int[med_id, 'spike', i, int_id, 'spike', j, 0, 'name'] = 'med_int_%s_%s' % (i, j)
    conn_med_int[med_id, 'spike', i, int_id, 'spike', j, 0, 'type'] = 'AlphaSynapse'

conn_med_int[med_id, 'spike', sl_med, int_id, 'spike', sl_int, 0, 'reverse'] = -80.0
conn_med_int[med_id, 'spike', sl_med, int_id, 'spike', sl_int, 0, 'gmax'] = 50.0
conn_med_int[med_id, 'spike', sl_med, int_id, 'spike', sl_int, 0, 'conductance'] = 5.0
conn_med_int[med_id, 'spike', sl_med, int_id, 'spike', sl_int, 0, 'ar'] = 1.0
conn_med_int[med_id, 'spike', sl_med, int_id, 'spike', sl_int, 0, 'ad'] = 200.0
conn_med_int[med_id, 'spike', sl_med, int_id, 'spike', sl_int, 0, 'class'] = 0

man.connect(lpu_med, lpu_int, conn_med_int)

# The projection neurons are the very last neurons in the total vector of
# antennal lobe neurons:
conn_al_int = core.Connectivity(0, N_al, 0, N_int, 1,
                                 al_id, int_id, alphasynapse_type_params)
sl_al = slice(N_al-N_al_pn, N_al)
for id, (i, j) in enumerate(itertools.product(xrange(N_al-N_al_pn, N_al),
                                              xrange(N_int))):
    conn_al_int[al_id, 'spike', i, int_id, 'spike', j] = 1
    conn_al_int[al_id, 'spike', i, int_id, 'spike', j, 0, 'id'] = id
    conn_al_int[al_id, 'spike', i, int_id, 'spike', j, 0, 'name'] = 'al_int_%s_%s' % (i, j)
    conn_al_int[al_id, 'spike', i, int_id, 'spike', j, 0, 'type'] = 'AlphaSynapse'

conn_al_int[al_id, 'spike', sl_al, int_id, 'spike', sl_int, 0, 'reverse'] = -80.0
conn_al_int[al_id, 'spike', sl_al, int_id, 'spike', sl_int, 0, 'gmax'] = 50.0
conn_al_int[al_id, 'spike', sl_al, int_id, 'spike', sl_int, 0, 'conductance'] = 5.0
conn_al_int[al_id, 'spike', sl_al, int_id, 'spike', sl_int, 0, 'ar'] = 1.0
conn_al_int[al_id, 'spike', sl_al, int_id, 'spike', sl_int, 0, 'ad'] = 300.0
conn_al_int[al_id, 'spike', sl_al, int_id, 'spike', sl_int, 0, 'class'] = 0

man.connect(lpu_al, lpu_int, conn_al_int)

man.start(steps=args.steps)
man.stop()
