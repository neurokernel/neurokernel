import numpy as np
import h5py
import networkx as nx
import argparse
import itertools
import random
import pickle
from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.tools.logging import setup_logger
from neurokernel.LPU.LPU import LPU

def main():
    import neurokernel.mpi_relaunch
    import neurokernel.core_gpu as core
    (comp_dict, conns) = LPU.lpu_parser('neuroballad_temp_model.gexf.gz')
    with open('run_parameters.pickle', 'rb') as f:
        run_parameters = pickle.load(f)
    dur = 1.0
    dt = 1e-4
    dur = run_parameters[0]
    dt = run_parameters[1]
    fl_input_processor = FileInputProcessor('neuroballad_temp_model_input.h5')

    from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
    output_processor = FileOutputProcessor([('V',None),('spike_state',None),('I',None)], 'neuroballad_temp_model_output.h5', sample_interval=1)

    #Parse extra arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False,
                        dest='debug', action='store_true',
                        help='Write connectivity structures and inter-LPU routed data in debug folder')
    parser.add_argument('-l', '--log', default='file', type=str,
                        help='Log output to screen [file, screen, both, or none; default:none]')
    parser.add_argument('-r', '--time_sync', default=False, action='store_true',
                        help='Time data reception throughput [default: False]')
    parser.add_argument('-g', '--gpu_dev', default=[0, 1], type=int, nargs='+',
                        help='GPU device numbers [default: 0 1]')
    parser.add_argument('-d', '--disconnect', default=False, action='store_true',
                        help='Run with disconnected LPUs [default: False]')
    args = parser.parse_args()
    file_name = None
    screen = False
    if args.log.lower() in ['file', 'both']:
        file_name = 'neurokernel.log'
    if args.log.lower() in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)
    man = core.Manager()

    man.add(LPU, 'lpu', dt, comp_dict, conns,
            input_processors = [fl_input_processor ],
            output_processors = [output_processor], device=args.gpu_dev[1],
                    debug=True)

    steps = int(dur/dt)
    man.spawn()
    man.start(steps = steps)
    man.wait()

if __name__=='__main__':
    main()
