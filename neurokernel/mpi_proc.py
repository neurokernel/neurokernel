#!/usr/bin/env python

"""
Classes for managing MPI-based processes.
"""

import inspect
import os
import sys
from .routing_table import RoutingTable

# Use dill for mpi4py object serialization to accomodate a wider range of argument
# possibilities than possible with pickle:
import dill
from future.utils import iteritems

# Fix for bug https://github.com/uqfoundation/dill/issues/81
@dill.register(property)
def save_property(pickler, obj):
    pickler.save_reduce(property, (obj.fget, obj.fset, obj.fdel), obj=obj)

import twiggy
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
from mpi4py.MPI import Info

# mpi4py has changed the method to override pickle with dill various times
try:
    # mpi4py 3.0.0
    MPI.pickle.__init__(dill.dumps, dill.loads)
except AttributeError:
    try:
        # mpi4py versions 1.3.1 through 2.x
        MPI.pickle.dumps = dill.dumps
        MPI.pickle.loads = dill.loads
    except AttributeError:
        # mpi4py pre 1.3.1
        MPI._p_pickle.dumps = dill.dumps
        MPI._p_pickle.loads = dill.loads

from .mixins import LoggerMixin
from .tools.logging import set_excepthook
from .tools.misc import memoized_property
from .all_global_vars import all_global_vars

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

    if sys.version_info.major == 2:
        spec = inspect.getargspec(f)
    else:
        spec = inspect.getfullargspec(f)
    return spec.args[1:] if inspect.ismethod(f) or 'self' in spec.args else spec.args


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
    assert len(arg_names) >= len(args)
    for arg, val in zip(arg_names, args):
        d[arg] = val
    for arg, val in iteritems(kwargs):
        if arg in d:
            raise ValueError('\'%s\' already specified in positional args' % arg)
        d[arg] = val
    return d

class Process(LoggerMixin):
    """
    Process class.

    Parameters
    ----------
    manager: bool
        Managerless running mode flag. It False, run Module without a
        manager. (default: True).
    """

    def __init__(self, manager = True, *args, **kwargs):
        LoggerMixin.__init__(self, 'prc %s' % MPI.COMM_WORLD.Get_rank() if manager else 0)
        set_excepthook(self.logger, True)

        self._args = args
        self._kwargs = kwargs
        self.manager = manager

    @memoized_property
    def intracomm(self):
        """
        Intracommunicator to access peer processes.
        """

        return MPI.COMM_WORLD if self.manager else None

    @memoized_property
    def intercomm(self):
        """
        Intercommunicator to access parent process.
        """

        return MPI.Comm.Get_parent() if self.manager else None

    @memoized_property
    def rank(self):
        """
        MPI process rank.
        """

        return MPI.COMM_WORLD.Get_rank() if self.manager else 0

    @memoized_property
    def size(self):
        """
        Number of peer processes.
        """

        return MPI.COMM_WORLD.Get_size() if self.manager else 1

    def run(self, steps = 0):
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

    def spawn(self, **kwargs):
        """
        Spawn MPI processes for and execute each of the managed targets.

        Parameters
        ----------
        kwargs: dict
                options for the `info` argument in mpi spawn process.
                see https://www.open-mpi.org/doc/v4.0/man3/MPI_Comm_spawn.3.php
        """

        # Typcially MPI must be have intialized before spawning.
        if not MPI.Is_initialized():
            MPI.Init()

        if self._is_parent:
            # Find the path to the mpi_backend.py script (which should be in the
            # same directory as this module:
            parent_dir = os.path.dirname(__file__)
            mpi_backend_path = os.path.join(parent_dir, 'mpi_backend.py')

            # Set spawn option. Due to --oversubscribe, we will use none in binding
            info = Info.Create()
            info.Set('bind_to', "none")

            for k, v in kwargs.items():
                info.Set(k, v)

            # Spawn processes:
            self._intercomm = MPI.COMM_SELF.Spawn(sys.executable,
                                            args=[mpi_backend_path],
                                            maxprocs=len(self),
                                            info = info)

            # First, transmit twiggy logging emitters to spawned processes so
            # that they can configure their logging facilities:
            for i in self._targets:
                self._intercomm.send(twiggy.emitters, i)

            # Next, serialize the routing table ONCE and then transmit it to all
            # of the child nodes:
            try:
                routing_table = self.routing_table
            except:
                routing_table = RoutingTable()
                self.log_warning('Routing Table is null, using empty routing table.')

            self._intercomm.bcast(routing_table, root=MPI.ROOT)

            # Transmit class to instantiate, globals required by the class, and
            # the constructor arguments; the backend will wait to receive
            # them and then start running the targets on the appropriate nodes.
            req = MPI.Request()
            r_list = []
            for i in self._targets:
                target_globals = all_global_vars(self._targets[i])

                # Serializing atexit with dill appears to fail in virtualenvs
                # sometimes if atexit._exithandlers contains an unserializable function:
                if 'atexit' in target_globals:
                    del target_globals['atexit']
                data = (self._targets[i], target_globals, self._kwargs[i])
                r_list.append(self._intercomm.isend(data, i))

                # Need to clobber data to prevent all_global_vars from
                # including it in its output:
                del data
            req.Waitall(r_list)

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
