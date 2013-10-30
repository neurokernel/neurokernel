"""
A script for testing the revision of LPU module
"""

import neurokernel.core as core
import neurokernel.base as base
import neurokernel.tools.graph as graph_tools
import networkx as nx

from neurokernel.LPU.lpu_parser import lpu_parser
from neurokernel.LPU.LPU import LPU
from neurokernel.LPU.LPU_rev import LPU_rev
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug',dest='debug',action='store_true',
            help = ("Write connectivity structures and inter_lpu routed data in debug folder"))
parser.set_defaults(debug=False)
parser.add_argument('-s', '--steps', help = ('No of steps [default:10000]'))
parser.add_argument('-d', '--data_port', help = ('Data port [default:5005]'))
parser.add_argument('-c', '--ctrl_port', help = ('Control port [default:5006]'))
parser.add_argument('-a', '--al_device', help = ('GPU to use for antennal lobe [default:4]'))
args = parser.parse_args()


if args.data_port:
    data_port = int(args.data_port)
else:
    data_port = 5005

if args.ctrl_port:
    ctrl_port = int(args.ctrl_port)
else:
    ctrl_port = 5006

if args.al_device:
    dev1 = int(args.al_device)
else:
    dev1 = 4

if args.steps:
    steps= int(args.steps)
else:
    steps=10000

dt = 1e-4
dur = 1.0
Nt = 10000#int(dur/dt)

logger = base.setup_logger()

man = core.Manager(port_data=data_port, port_ctrl=ctrl_port)
man.add_brok()


(n_dict, s_dict) = LPU_rev.lpu_parser( './config_files/antennallobe.gexf.gz')
#(n_dict, s_dict) = LPU_rev.lpu_parser( './config_files/antennallobe_no_synapse.gexf.gz')


al = LPU_rev( dt, n_dict, s_dict,\
        #input_file='videos/flciker_stripe_same6.h5',
                    output_file='al_output.h5', port_ctrl=man.port_ctrl,
                    port_data=man.port_data, device=dev1, id='a;',
                    debug=args.debug)

al = man.add_mod(al)


man.start(steps=step)
man.stop()

'''
The extra step is required as during the first step,
only the initial states are passed between the modules.
'''
