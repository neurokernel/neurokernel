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

    # Retry without redirection if an IOError occurs, e.g., in an IPython
    # notebook because the overriden iostreams don't have a file
    # descriptor:
    try:
        subprocess.call(['mpiexec', '-np', '1',
                        sys.executable, script_name]+sys.argv[1:],
                        env=env,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        stdin=sys.stdin)
    except IOError:
        try:
            subprocess.call(['mpiexec', '-np', '1',
                            sys.executable, script_name]+sys.argv[1:],
                            env=env)
        except Exception as e:
            raise RuntimeError('cannot execute mpiexec: %s' % str(e.message))
        else:
            sys.exit(0)
    else:
        sys.exit(0)

