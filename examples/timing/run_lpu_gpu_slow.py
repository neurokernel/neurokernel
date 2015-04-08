#!/usr/bin/env python

"""
Run timing test (GPU) scaled over number of LPUs.
"""

import csv
import glob
import os
import re
import subprocess
import sys

import numpy as np

out_file = sys.argv[1]
script_name = 'timing_demo_gpu_slow.py'
trials = 3

f = open(out_file, 'w', 0)
w = csv.writer(f)
for spikes in xrange(250, 7000, 250):
    for lpus in xrange(2, 9):
        for i in xrange(trials):
            # CUDA < 7.0 doesn't properly clean up IPC-related files; since
            # these can cause problems, we manually remove them before launching
            # each job:
            ipc_files = glob.glob('/dev/shm/cuda.shm*')
            for ipc_file in ipc_files:
                os.remove(ipc_file)
            out = subprocess.check_output(['srun', '-n', '1', '-c', str(lpus),
                                           '--gres=gpu:%i' % lpus,
                                           '-p', 'huxley',
                                           'python', script_name,
                                           '-u', str(lpus), '-s', str(spikes/(lpus-1)),
                                           '-g', '0', '-m', '50'])
            average_step_sync_time, runtime_all, runtime_main, \
                runtime_loop = out.strip('()\n\"').split(', ')
            w.writerow([lpus, spikes, average_step_sync_time,
                        runtime_all, runtime_main, runtime_loop])
f.close()
