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
env_keys = list(env.keys())
for k in env_keys:
    if k.startswith('SLURM'):
        del env[k]

# Additional options to pass to mpiexec for debugging purposes; some
# useful flags for debugging CUDA issues are
# MPIEXEC_EXTRA_OPTS = ['--mca', 'mpi_common_cuda_verbose', '200',
#                       '--mca', 'mpool_rgpusm_verbose', '100',
#                       '--mca', 'mpi_common_cuda_gpu_mem_check_workaround', '0']
# To prevent OpenMPI from being confused by virtual interfaces, one can
# explicitly specify which interfaces to use (e.g., eth0) using
# MPIEXEC_EXTRA_OPTS = ['--mca', 'btl_tcp_if_include', 'eth0']
MPIEXEC_EXTRA_OPTS = []

# Get name of the file in which this module is imported:
script_name = inspect.stack()[-1][1]
parent_name = psutil.Process(os.getppid()).name()
if not re.search('mpirun|mpiexec', parent_name):

    # Retry without redirection if an IOError occurs, e.g., in an IPython
    # notebook because the overriden iostreams don't have a file
    # descriptor:
    try:
        subprocess.call(['mpiexec', '--oversubscribe', '--bind-to', 'none', '-np', '1']+MPIEXEC_EXTRA_OPTS+\
                        [sys.executable, script_name]+sys.argv[1:],
                        env=env,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        stdin=sys.stdin)
    except IOError:
        try:
            subprocess.call(['mpiexec', '--oversubscribe', '--bind-to', 'none', '-np', '1',
                            sys.executable, script_name]+sys.argv[1:],
                            env=env)
        except Exception as e:
            raise RuntimeError('cannot execute mpiexec: %s' % str(e.message))
        else:
            sys.exit(0)
    else:
        sys.exit(0)
