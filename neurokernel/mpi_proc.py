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

    def send_parent(self, data, tag=0):
        """
        Send data to parent process.
        """

        self.intercomm.send(data, 0, tag=tag)

    def recv_parent(self, tag=MPI.ANY_TAG):
        """
        Receive data from parent process.
        """

        return self.intercomm.recv(tag=tag)

    def send_peer(self, data, dest, tag=0):
        """
        Send data to peer process.
        """

        self.intracomm.send(data, dest, tag=tag)

    def recv_peer(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
        return self.intracomm.recv(source=source, tag=tag)

class ProcessManager(object):
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
        """
        Add target class or function to manager.

        Parameters
        ----------
        target : Process or function
            Class or function to instantiate in MPI process. If a class is
            specified, the `run()` method of the class will be executed in the
            process.
        args : sequence
            Sequential arguments to pass to target class constructor.
        kwargs : dict
            Named arguments to pass to target class constructor.
        """

        self._targets.append(target)
        self._args.append(args)
        self._kwargs.append(kwargs)

    def __len__(self):
        return len(self._targets)

    @memoized_property
    def _is_parent(self):
        """
        True if the current MPI process is the spawning parent.
        """

        return MPI.Comm.Get_parent() == MPI.COMM_NULL

    def run(self):
        """
        Spawn MPI processes for and execute each of the managed targets.
        """

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

    def send(self, data, dest, tag=0):
        """
        Send data to child process.
        """

        self.intercomm.send(data, dest, tag=0)

    def recv(self, tag=MPI.ANY_TAG):
        """
        Receive data from child process.
        """

        return self.intercomm.recv(tag=tag)
