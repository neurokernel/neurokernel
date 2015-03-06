#!/usr/bin/env python

"""
Run timing test
"""

import re
import subprocess

script_name = 'timing_demo_gpu.py'

for spikes in xrange(100, 1100, 100):
    out = subprocess.check_output(['python', script_name,
                    '-u', '2', '-s', str(spikes), '-g', '0', '-m', '1000'])
    throughput, runtime = out.strip().split(',')
    print spikes, throughput, runtime
