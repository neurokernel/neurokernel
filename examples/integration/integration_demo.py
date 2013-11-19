#!/usr/bin/python env

"""
Integration of two antennal lobe models.

Notes
-----
Generate input file by running ./data/gen_olf_input.py
"""

import argparse
import itertools

import pandas
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
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=10000, type=int,
                    help='Number of steps [default:10000]')
parser.add_argument('-d', '--data_port', default=5005, type=int,
                    help='Data port [default:5005]')
parser.add_argument('-c', '--ctrl_port', default=5006, type=int,
                    help='Control port [default:5006]')
parser.add_argument('-al', '--al_left_dev', default=0, type=int,
                    help='GPU for left antennal lobe [default:0]')
parser.add_argument('-ar', '--al_right_dev', default=1, type=int,
                    help='GPU for right antennal lobe [default:1]')
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

n_data = pandas.DataFrame(n_dict['LeakyIAF'])
s_data = pandas.DataFrame(s_dict['AlphaSynapse'])

al_left = LPU_rev(dt, n_dict, s_dict, input_file='./data/olfactory_input.h5',
                  output_file='olfactory_left_output.h5', port_ctrl=man.port_ctrl,
                  port_data=man.port_data,
                  device=args.al_left_dev, id='al_left',
                  debug=args.debug)
al_right = LPU_rev(dt, n_dict, s_dict, input_file='./data/olfactory_input.h5',
                   output_file='olfactory_right_output.h5', port_ctrl=man.port_ctrl,
                   port_data=man.port_data,
                   device=args.al_right_dev, id='al_right',
                   debug=args.debug)
man.add_mod(al_left)
man.add_mod(al_right)

# Find V.* projection neuron indices:
V_pn_ind = n_data[n_data['name'].str.contains('V.*pn.*')].index.astype(int)
N = len(n_data)

# Create one-way connections from all V.* PNs on the left to all of those on the
# right:
conn = core.Connectivity(0, N, 0, N, 1, al_left.id, al_right.id)
for id, (i, j) in enumerate(itertools.product(V_pn_ind, V_pn_ind)):

    name = 'syn_%s_%s' % (i, j)
    
    # Set connection:
    conn[al_left.id, 'spike', i, al_right.id, 'spike', j] = 1

    # Set connection id and name:
    conn[al_left.id, 'spike', i, al_right.id, 'spike', j, 0, 'id'] = id
    conn[al_left.id, 'spike', i, al_right.id, 'spike', j, 0, 'name'] = name
    
    # Set connection type:
    conn[al_left.id, 'spike', i, al_right.id, 'spike', j, 0, 'type'] = 'AlphaSynapse'

# Set parameters:
sl = slice(min(V_pn_ind), max(V_pn_ind))
conn[al_left.id, 'spike', sl, al_right.id, 'spike', sl, 0, 'reverse'] = -80.0
conn[al_left.id, 'spike', sl, al_right.id, 'spike', sl, 0, 'gmax'] = 10.0
conn[al_left.id, 'spike', sl, al_right.id, 'spike', sl, 0, 'conductance'] = 1.0
conn[al_left.id, 'spike', sl, al_right.id, 'spike', sl, 0, 'ar'] = 1.0
conn[al_left.id, 'spike', sl, al_right.id, 'spike', sl, 0, 'ad'] = 300.0
conn[al_left.id, 'spike', sl, al_right.id, 'spike', sl, 0, 'class'] = 0

man.connect(al_left, al_right, conn)
man.start(steps=args.steps)
man.stop()
