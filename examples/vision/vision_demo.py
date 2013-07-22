import neurokernel.core as core
import neurokernel.base as base
import neurokernel.tools.graph as graph_tools
import networkx as nx

from neurokernel.LPU.lpu_parser import lpu_parser
from neurokernel.LPU.LPU import LPU
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug',dest='debug',action='store_true',
            help = ("Write connectivity structures and inter_lpu routed data in debug folder"))
parser.set_defaults(debug=False)
parser.add_argument('-s', '--steps', help = ('No of steps [default:10000]'))
parser.add_argument('-d', '--data_port', help = ('Data port [default:5005]'))
parser.add_argument('-c', '--ctrl_port', help = ('Control port [default:5006]'))
parser.add_argument('-l', '--lamina_device', help = ('GPU to use for lamina [default:0]'))
parser.add_argument('-m', '--medulla_device', help = ('GPU to use for medulla [default:1]'))
args = parser.parse_args()


if args.data_port:
    data_port = int(args.data_port)
else:
    data_port = 5005
    
if args.ctrl_port:
    ctrl_port = int(args.ctrl_port)
else:
    ctrl_port = 5006

if args.lamina_device:
    dev1 = int(args.lamina_device)
else:
    dev1 = 0

if args.medulla_device:
    dev2 = int(args.medulla_device)
else:
    dev2 = 1


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

(n_dict_lam, s_dict_lam) = lpu_parser('./config_files/lamina.gexf')
lam = LPU( dt, n_dict_lam, s_dict_lam,
                    input_file='videos/flicker_stripe_same6.h5',
                    output_file='lamina_output.h5', port_ctrl= man.port_ctrl,
                    port_data=man.port_data, device=dev1, LPU_id='lamina', debug=args.debug)
print 'lamina init done'
# initialize medulla

'''
(n_dict_med, s_dict_med) = lpu_parser('./config_files/medulla.gexf')
med = LPU(dt, n_dict_med, s_dict_med,
                    output_file='medulla_output.h5', port_ctrl= man.port_ctrl,
                    port_data=man.port_data, device=dev2, LPU_id='medulla',debug=args.debug)
print 'medulla init done'

'''
lam = man.add_mod(lam)
'''
med = man.add_mod(med)

graph = nx.read_gexf('./config_files/lamina_medulla.gexf', relabel=True)
lam_med_conn = graph_tools.graph_to_conn(graph)

man.connect(lam, med, lam_med_conn)
'''
man.start(steps=10001)
man.join_modules()
man.stop_brokers()
'''
The extra step is required as during the first step,
only the initial states are passed between the modules.
'''
