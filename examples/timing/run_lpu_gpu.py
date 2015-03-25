#!/usr/bin/env python

"""
Run timing test (GPU) scaled over number of LPUs.
"""

import numpy as np

import csv
import re
import subprocess
import sys

script_name = 'timing_demo_gpu.py'

w = csv.writer(sys.stdout)
total_spike = 1000
for lpus in xrange(2, 9):
    average_step_sync_time_list = []
    average_throughput_list = []
    total_throughput_list = []
    runtime_all_list = []
    runtime_main_list = []
    runtime_loop_list = []
    for i in xrange(3):
        out = subprocess.check_output(['srun', '-n', '1', '-c', str(lpus),
                                       '--gres=gpu:%i' % lpus,
                                       'python', script_name,
                                       '-u', str(lpus),
                                       '-s', str(total_spike/(lpus-1)), '-g', '0',
                                       '-m', '50'])
        average_step_sync_time, average_throughput, total_throughput, \
            runtime_all, runtime_main, runtime_loop = out.strip('()\n\"').split(', ')
        average_step_sync_time_list.append(float(average_step_sync_time))
        average_throughput_list.append(float(average_throughput))
        total_throughput_list.append(float(total_throughput))
        runtime_all_list.append(float(runtime_all))
        runtime_main_list.append(float(runtime_main))
        runtime_loop_list.append(float(runtime_loop))
    w.writerow([lpus,
                np.average(average_step_sync_time_list),
                np.average(average_throughput_list),
                np.average(total_throughput_list),
                np.average(runtime_all_list),
                np.average(runtime_main_list),
                np.average(runtime_loop_list)])
