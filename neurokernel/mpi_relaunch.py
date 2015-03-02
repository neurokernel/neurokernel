#!/usr/bin/env python

"""
Relaunches an mpi4py program with mpiexec so that it can spawn processes.
Should be the first import in the program.
"""

import inspect
import os
import re
import subprocess
import sys

from mpi4py import MPI
import psutil

# Prevent SLURM from preventing mpiexec from starting multiple processes
# (this approach should probably be modified to take some SLURM variables into
# consideration):
env = os.environ.copy()
for k in env.keys():
    if k.startswith('SLURM'):
        del env[k]

# Get name of the file in which this module is imported:
script_name = inspect.stack()[1][1]
parent_name = psutil.Process(os.getppid()).name()
if not re.search('mpirun|mpiexec', parent_name):
    try:
        subprocess.call(['mpiexec', '-np', '1',
                        sys.executable, script_name],
                        env=env,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        stdin=sys.stdin)
    except IOError:
        raise IOError('cannot execute mpiexec')
    else:
        sys.exit(0)

