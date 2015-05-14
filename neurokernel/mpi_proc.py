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

# Fix for bug https://github.com/uqfoundation/dill/issues/81
@dill.register(property)
def save_property(pickler, obj):
    pickler.save_reduce(property, (obj.fget, obj.fset, obj.fdel), obj=obj)

import twiggy
from mpi4py import MPI

# The MPI._p_pickle attribute in the stable release of mpi4py 1.3.1
# was renamed to pickle in subsequent dev revisions:
try:
    MPI.pickle.dumps = dill.dumps
    MPI.pickle.loads = dill.loads
except AttributeError:
    MPI._p_pickle.dumps = dill.dumps
    MPI._p_pickle.loads = dill.loads

from mixins import LoggerMixin
from tools.logging import set_excepthook
from tools.misc import memoized_property
from all_global_vars import all_global_vars

def getargnames(f):
    """
    Get names of a callable's arguments.

    Parameters
    ----------
    f : callable
        Function to examine.

    Results
    -------
    args : list of str
        Argument names.

    Notes
    -----
    For instance methods, the `self` argument is omitted.
    """

    spec = inspect.getargspec(f)
    if inspect.ismethod(f):
        return spec.args[1:]
    else:
        return spec.args

def args_to_dict(f, *args, **kwargs):
    """
    Combine sequential and named arguments in single dictionary.

    Parameters
    ----------
    f : callable
        Function to which the arguments will be passed.
    args : tuple
        Sequential arguments.
    kwargs : dict
        Named arguments.

    Returns
    -------
    d : dict
        Maps argument names to values.
    """

    d = {}

    arg_names = getargnames(f)
    assert len(arg_names) <= args
    for arg, val in zip(arg_names, args):
        d[arg] = val
    for arg, val in kwargs.iteritems():
        if arg in d:
            raise ValueError('\'%s\' already specified in positional args' % arg)
        d[arg] = val
    return d

class Process(LoggerMixin):
    """
    Process class.
    """

    def __init__(self, *args, **kwargs):        
        LoggerMixin.__init__(self, 'prc %s' % MPI.COMM_WORLD.Get_rank())
        set_excepthook(self.logger, True)

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

class ProcessManager(LoggerMixin):
    """
    Process manager class.
    """

    def __init__(self):
        LoggerMixin.__init__(self, 'man')
        set_excepthook(self.logger, True)

        self._targets = {}
        self._args = {}
        self._kwargs = {}
        self._intercomm = MPI.COMM_NULL

        self._rank = 0

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
        target : Process
            Class instantiate and run in MPI process. 
        args : sequence
            Sequential arguments to pass to target class constructor.
        kwargs : dict
            Named arguments to pass to target class constructor.

        Returns
        -------
        rank : int
            MPI rank assigned to class. Targets are assigned ranks starting with
            0 in the order that they are added.
        """

        assert issubclass(target, Process)
        rank = self._rank
        self._targets[rank] = target
        self._kwargs[rank] = args_to_dict(target.__init__, *args, **kwargs)
        self._rank += 1
        return rank

    def __len__(self):
        return len(self._targets)

    @memoized_property
    def _is_parent(self):
        """
        True if the current MPI process is the spawning parent.
        """

        return MPI.Comm.Get_parent() == MPI.COMM_NULL

    def spawn(self):
        """
        Spawn MPI processes for and execute each of the managed targets.
        """

        if self._is_parent:
            # Find the path to the mpi_backend.py script (which should be in the
            # same directory as this module:
            parent_dir = os.path.dirname(__file__)
            mpi_backend_path = os.path.join(parent_dir, 'mpi_backend.py')

            # Spawn processes:
            self._intercomm = MPI.COMM_SELF.Spawn(sys.executable,
                                            args=[mpi_backend_path],
                                            maxprocs=len(self))

            # First, transmit twiggy logging emitters to spawned processes so
            # that they can configure their logging facilities:
            for i in self._targets.keys():
                self._intercomm.send(twiggy.emitters, i)

            # Transmit class to instantiate, globals required by the class, and
            # the constructor arguments; the backend will wait to receive
            # them and then start running the targets on the appropriate nodes.
            for i in self._targets.keys():
                target_globals = all_global_vars(self._targets[i])

                # Serializing atexit with dill appears to fail in virtualenvs
                # sometimes if atexit._exithandlers contains an unserializable function:
                if 'atexit' in target_globals:
                    del target_globals['atexit']
                data = (self._targets[i], target_globals, self._kwargs[i])
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

if __name__ == '__main__':
    import mpi_relaunch

    class MyProcess(Process):
        def __init__(self, *args, **kwargs):
            super(MyProcess, self).__init__(*args, **kwargs)
            self.log_info('I am process %d of %d on %s.' % \
                          (MPI.COMM_WORLD.Get_rank(),
                           MPI.COMM_WORLD.Get_size(),
                           MPI.COMM_WORLD.Get_name()))
        
    from tools.logging import setup_logger

    setup_logger(screen=True, multiline=True)
    
    man = ProcessManager()
    man.add(MyProcess, 1, 2, a=3)
    man.add(MyProcess, 4, b=5, c=6)
    man.spawn()
