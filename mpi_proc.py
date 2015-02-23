#!/usr/bin/env python

"""
Classes for managing MPI-based processes.
"""

import inspect
import sys

import dill
from mpi4py import MPI

class MPIProcess(object):
    """
    Process class.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.comm = MPI.COMM_WORLD
        self.parent = MPI.Comm.Get_parent()
        self.rank = self.comm.Get_rank()

    def run(self):
        pass

class MPIProcMan(object):
    """
    Process manager class.
    """

    def __init__(self):
        self._targets = []
        self._args = []
        self._kwargs = []
        self.comm = MPI.COMM_WORLD
        self.parent = MPI.Comm.Get_parent()
        self.rank = self.comm.Get_rank()

    def add(self, target, *args, **kwargs):
        self._targets.append(target)
        self._args.append(args)
        self._kwargs.append(kwargs)

    def __len__(self):
        return len(self._targets)

    def _is_parent(self):
        return self.parent == MPI.COMM_NULL

    def run(self):
        if self._is_parent():
            # Find the file name of the module in which MPIProcMan 
            # is instantiated:
            file_name = inspect.stack()[1][1]

            # Spawn processes:
            self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                                            args=['mpi_backend.py', file_name],
                                            maxprocs=len(self))

            # Transmit class name, args, and kwargs; the
            # backend will wait to receive them and then start running the
            # targets on the appropriate nodes.
            for i in xrange(len(self)):
                data = dill.dumps((self._targets[i].__name__,
                                   self._args[i], self._kwargs[i]))
                self.comm.send(data, i)

    def send(self, data, dest):
        if self._is_parent():
            if dest >= len(self):
                raise ValueError('nonexistent destination')
            self.comm.send(data, dest)

    def recv(self):
        if self._is_parent():
            return self.comm.recv()

