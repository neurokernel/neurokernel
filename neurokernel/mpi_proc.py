#!/usr/bin/env python

"""
Classes for managing MPI-based processes.
"""

import inspect
import os
import sys

# Use dill for mpi4py object serialization to accomodate a wider range of argument
# possibilities than possible with pickle:
import dill
from mpi4py import MPI
MPI.pickle.dumps = dill.dumps
MPI.pickle.loads = dill.loads

from tools.misc import memoized_property

class Process(object):
    """
    Process class.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @memoized_property
    def intracomm(self):
        """
        Intracommunicator to access peer processes.
        """

        return MPI.COMM_WORLD

    @memoized_property
    def intercomm(self):
        """
        Intercommunicator to access parent process.
        """

        return MPI.Comm.Get_parent()

    @memoized_property
    def rank(self):
        """
        MPI process rank.
        """

        return MPI.COMM_WORLD.Get_rank()

    @memoized_property
    def size(self):
        """
        Number of peer processes.
        """

        return MPI.COMM_WORLD.Get_size()

    def run(self):
        """
        Process body.
        """

        pass

    def send_parent(self, data):
        self.intercomm.send(data)

    def recv_parent(self):
        return self.intercomm.recv()

    def send_peer(self, data):
        self.intracomm.send(data)

    def recv_peer(self):
        return self.intracomm.recv()

class Manager(object):
    """
    Process manager class.
    """

    def __init__(self):
        self._targets = []
        self._args = []
        self._kwargs = []
        self._intercomm = MPI.COMM_NULL

    @property
    def intercomm(self):
        """
        Intercommunicator to spawned processes.

        Notes
        -----
        Set to COMM_NULL until the run() method is called.
        """

        return self._intercomm

    def add(self, target, *args, **kwargs):
        self._targets.append(target)
        self._args.append(args)
        self._kwargs.append(kwargs)

    def __len__(self):
        return len(self._targets)

    @memoized_property
    def _is_parent(self):
        return MPI.Comm.Get_parent() == MPI.COMM_NULL

    def run(self):
        if self._is_parent:
            # Find the file name of the module in which the Process class
            # is instantiated:
            file_name = inspect.stack()[1][1]

            # Find the path to the mpi_backend.py script (which should be in the
            # same directory as this module:
            parent_dir = os.path.dirname(__file__)
            mpi_backend_path = os.path.join(parent_dir, 'mpi_backend.py')

            # Spawn processes:
            self._intercomm = MPI.COMM_SELF.Spawn(sys.executable,
                                            args=[mpi_backend_path, file_name],
                                            maxprocs=len(self))

            # Transmit class name, args, and kwargs; the
            # backend will wait to receive them and then start running the
            # targets on the appropriate nodes.
            for i in xrange(len(self)):
                data = (self._targets[i].__name__, self._args[i], self._kwargs[i])
                self._intercomm.send(data, i)

    def send(self, data, i):
        self.intercomm.send(data, i)

    def recv(self):
        return self.intercomm.recv()
