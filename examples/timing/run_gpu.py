#!/usr/bin/env python

"""
Run timing test (GPU) scaled over number of ports.
"""

import csv
import re
import subprocess
import sys

import numpy as np

script_name = 'timing_demo_gpu.py'
trials = 3
lpus = 2

w = csv.writer(sys.stdout)
for spikes in np.linspace(500, 15000, 20, dtype=int):
    for i in xrange(trials):
        out = subprocess.check_output(['srun', '-n', '1', '-c', str(lpus+2),
                                       '--gres=gpu:%s' % lpus, '-p', 'huxley',
                                       'python', script_name,
                                       '-u', str(lpus), '-s', str(spikes),
                                       '-g', '0', '-m', '50'])
        average_step_sync_time, runtime_all, runtime_main, \
            runtime_loop = out.strip('()\n\"').split(', ')
        w.writerow([lpus, spikes, average_step_sync_time,             
                    runtime_all, runtime_main, runtime_loop])
