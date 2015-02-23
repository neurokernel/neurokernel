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

# Get name of the file in which this module is imported:
script_name = os.path.basename(inspect.stack()[1][1])
parent_name = psutil.Process(os.getppid()).name()
if not re.search('mpirun|mpiexec', parent_name):
    subprocess.call(['mpiexec', '-np', '1',
                    sys.executable, script_name],
                    env=os.environ,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    stdin=sys.stdin)
    sys.exit(0)

