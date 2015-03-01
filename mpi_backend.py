#!/usr/bin/env python

"""
Backend program invoked by MPI spawn.
"""

import importlib
import inspect
import os.path
import sys

import dill
from mpi4py import MPI
MPI.pickle.dumps = dill.dumps
MPI.pickle.loads = dill.loads

import mpi_proc

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
parent = MPI.Comm.Get_parent()

# Import the module containing the various classes/functions that must be run:
mod_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
m = importlib.import_module(mod_name)

# Get the constructor/function arguments:
name, args, kwargs = parent.recv()

# Target is a class:
target = getattr(m, name)
if inspect.isclass(target) and issubclass(target, mpi_proc.Process):
    instance = target(*args, **kwargs)
    instance.run()

# Target is a function:
else:
    target(*args, **kwargs)
