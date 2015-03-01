#!/usr/bin/env python

"""
Demo of how to use write a self-launched MPI program that spawns processes.
"""

import mpi_relaunch

import sys

from mpi4py import MPI

if MPI.Comm.Get_parent() == MPI.COMM_NULL:
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=[__file__], maxprocs=3)
    print 'parent'
else:
    rank = MPI.COMM_WORLD.Get_rank()
    print 'child: %s' % rank
