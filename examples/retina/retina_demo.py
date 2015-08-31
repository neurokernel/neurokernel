#!/usr/bin/env python

import argparse
import time

from  neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core_gpu
from neurokernel.LPU.LPU import LPU

from data.gen_vis_input import generate_input
from data.gen_vis_gexf import generate_gexf

def main():
    import neurokernel.mpi_relaunch
    
    logger = setup_logger(file_name=None, screen=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', dest='num_layers', type=int,
                        default=16,
                        help='number of layers of ommatidia on circle')

    parser.add_argument('-i', '--input', action="store_true",
                        help='generates input if set')
    parser.add_argument('-g', '--gexf', action="store_true",
                        help='generates gexf of LPU if set')

    parser.add_argument('--steps', default=100, type=int,
                        help='simulation steps')

    args = parser.parse_args()

    dt = 1e-4
    GEXF_FILE = 'retina.gexf.gz'
    INPUT_FILE = 'vision_input.h5'
    IMAGE_FILE = 'image1.mat'
    OUTPUT_FILE = 'retina_output.h5'

    if args.input:
        print('Generating input of model from image file')
        generate_input(INPUT_FILE, IMAGE_FILE, args.num_layers)
    if args.gexf:
        print('Writing retina lpu')
        n = args.num_layers
        photoreceptor_num = 6*(3*n*(n+1)+1)
        generate_gexf(GEXF_FILE, photoreceptor_num)

    man = core_gpu.Manager()

    print('Parsing lpu data')
    n_dict_ret, s_dict_ret = LPU.lpu_parser(GEXF_FILE)
    print('Initializing LPU')
    man.add(LPU, 'retina', dt, n_dict_ret, s_dict_ret,
                  input_file=INPUT_FILE,
                  output_file=OUTPUT_FILE, 
                  device=0, 
                  debug=True, time_sync=False)
    man.spawn()

    print('Starting simulation')
    start_time = time.time()
    man.start(steps=args.steps)
    man.wait()
    print('Simulation complete: Duration {} seconds'.format(time.time() - 
                                                            start_time))

if __name__=='__main__':
    main()
