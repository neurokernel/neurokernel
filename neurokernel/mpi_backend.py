#!/usr/bin/env python

"""
Backend program invoked by MPI spawn.
"""

import importlib

# Use dill for mpi4py object serialization to accomodate a wider range of argument
# possibilities than possible with pickle:
import dill
from future.utils import iteritems

# XXX This is a Neuroarch-related workaround required to compensate for dill's
# inability to serialize namedtuple within a module:
try:
    import pyorient.ogm.graph
except ImportError:
    pass
else:
    setattr(pyorient.ogm.graph, 'orientdb_version',
            pyorient.ogm.graph.ServerVersion)

# Fix for bug https://github.com/uqfoundation/dill/issues/81
@dill.register(property)
def save_property(pickler, obj):
    pickler.save_reduce(property, (obj.fget, obj.fset, obj.fdel), obj=obj)

# Import atexit explicitly just in case a target class uses it because we can't
# serialize it properly in certain circumstances with dill 0.2.2; see
# https://github.com/uqfoundation/dill/issues/91
import atexit

import twiggy
from mpi4py import MPI

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

# This import must match the corresponding import in neurokernel.tools.logging
# so that the isinstance() check below for MPIOutput instances in transmitted
# emitters below can succeed; using a relative import in either place can cause
# the name of the class to not match:
import neurokernel.tools.mpi

# Process needs to be imported directly into the script's namespace in order to
# ensure that the issubclass() check later in the script succeeds:
from neurokernel.mpi_proc import Process
import neurokernel.mpi_proc

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
parent = MPI.Comm.Get_parent()

# Get emitters transmitted from spawning process:
emitters = parent.recv()

# If any of the emitters contain MPIOutput instances, they need to be replaced
# by newly initialized instances so that they write to valid file handles and
# use the intercommunicator to the parent process:
for k, v in iteritems(emitters):
    if isinstance(v._output, neurokernel.tools.mpi.MPIOutput):
        level = v.min_level
        name = v._output.filename
        format = v._output._format
        mode = v._output.mode

        # The close_atexit argument is explicitly set to False here because we need
        # to manually close the file handle associated with MPIOutput before
        # MPI.Finalize() is called via atexit in the base/core modules:
        twiggy.add_emitters(('file', level, None,
            neurokernel.tools.mpi.MPIOutput(name, format,
                                            MPI.COMM_WORLD, mode, False)))
    else:
        twiggy.emitters[k] = v

# Get the routing table:
routing_table =  parent.bcast(None, root=0)

# Get the target class/function and its constructor arguments:
target, target_globals, kwargs = parent.recv()

# Insert the transmitted globals into the current scope:
globals()[target.__name__] = target
for k, n in iteritems(target_globals):
    globals()[k] = n

# Add the routing table to the target arguments:
kwargs['routing_table'] = routing_table

# Instantiate and run the target class:
instance = target(**kwargs)
instance.run()
