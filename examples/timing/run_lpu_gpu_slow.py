#!/usr/bin/env python

"""
Run timing test (GPU) scaled over number of LPUs.
"""

import csv
import glob
import multiprocessing as mp
import os
import re
import subprocess
import sys

import numpy as np

from neurokernel.tools.misc import get_pids_open

try:
    from subprocess import DEVNULL
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

out_file = sys.argv[1]
script_name = 'timing_demo_gpu_slow.py'
trials = 3

def check_and_print_output(*args):
    try:
        out = subprocess.check_output(args[0], env=os.environ, stderr=DEVNULL)
    except Exception as e:
        out = e.output
    row = out.strip('[]\n\"').split(', ')
    row[1] = str(args[1])
    out = ','.join(row)
    print out
    return out

pool = mp.Pool(5)
results = []
results_spikes = []
for spikes in xrange(250, 7000, 250):
    for lpus in xrange(2, 9):
        for i in xrange(trials):
            # CUDA < 7.0 doesn't properly clean up IPC-related files; since
            # these can cause problems, we manually remove them before launching
            # each job:
            ipc_files = glob.glob('/dev/shm/cuda.shm*')
            for ipc_file in ipc_files:

                # Only remove files that are not being held open by any processes:
                if not get_pids_open(ipc_file):
                    os.remove(ipc_file)
            r = pool.apply_async(check_and_print_output,
                                 [['srun', '-n', '1', '-c', str(lpus),
                                   '--gres=gpu:%i' % lpus,
                                   '-p', 'huxley',
                                   'python', script_name,
                                   '-u', str(lpus), '-s', str(spikes/(lpus-1)),
                                   '-g', '0', '-m', '50'], spikes])
            results.append(r)
            results_spikes.append(spikes)
f = open(out_file, 'w', 0)
w = csv.writer(f)
for r, s in zip(results, results_spikes):

    # Include total number of spikes rather than per-LPU number of spikes in output:
    row = r.get().strip('[]\n\"').split(',')
    w.writerow(row)
f.close()
