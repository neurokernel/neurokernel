#!/usr/bin/env python

"""
Run timing test (GPU) scaled over number of LPUs.
"""

import csv
import re
import subprocess
import sys

import numpy as np

out_file = sys.argv[1]
script_name = 'timing_demo_gpu.py'
trials = 3

f = open(out_file, 'w', 0)
w = csv.writer(f)
for spikes in xrange(1000, 6000, 250):
    for lpus in xrange(2, 9):
        for i in xrange(trials):
            out = subprocess.check_output(['srun', '-n', '1', '-c', str(lpus),
                                           '--gres=gpu:%i' % lpus,
                                           'python', script_name,
                                           '-u', str(lpus), '-s', str(spikes/(lpus-1)),
                                           '-g', '0', '-m', '50'])
            average_step_sync_time, runtime_all, runtime_main, \
                runtime_loop = out.strip('()\n\"').split(', ')
            w.writerow([lpus, spikes, average_step_sync_time,
                        runtime_all, runtime_main, runtime_loop])
f.close()
