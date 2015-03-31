#!/usr/bin/env python

"""
Run timing test (GPU) scaled over number of ports.
"""

import csv
import glob
import os
import re
import subprocess
import sys

import numpy as np

out_file = sys.argv[1]
script_name = 'timing_demo_gpu.py'
trials = 3
lpus = 2

f = open(out_file, 'w', 0)
w = csv.writer(f)
for spikes in np.linspace(50, 15000, 25, dtype=int):
    for i in xrange(trials):
        # CUDA < 7.0 doesn't properly clean up IPC-related files; since
        # these can cause problems, we manually remove them before launching
        # each job:
        ipc_files = glob.glob('/dev/shm/cuda.shm*')
        for ipc_file in ipc_files:
            os.remove(ipc_file)
        out = subprocess.check_output(['srun', '-n', '1', '-c', str(lpus+2),
                                       '-p', 'huxley',
                                       '--gres=gpu:%s' % lpus,
                                       'python', script_name,
                                       '-u', str(lpus), '-s', str(spikes),
                                       '-g', '0', '-m', '50'])
        average_step_sync_time, runtime_all, runtime_main, \
            runtime_loop = out.strip('()\n\"').split(', ')
        w.writerow([lpus, spikes, average_step_sync_time,
                    runtime_all, runtime_main, runtime_loop])
f.close()
