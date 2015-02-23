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

import mpi_proc

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
parent = MPI.Comm.Get_parent()

# Import the module containing the various classes/functions that must be run:
mod_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
m = importlib.import_module(mod_name)

# Get the constructor/function arguments:
data = parent.recv()
name, args, kwargs = dill.loads(data)

# Target is a class:
target = getattr(m, name)
if inspect.isclass(target) and issubclass(target, mpi_proc.MPIProcess):
    instance = target(*args, **kwargs)
    instance.run()

# Target is a function:
else:
    target(*args, **kwargs)
