#!/usr/bin/env python

"""
MPI utilities.
"""

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
import twiggy

class MPIOutput(twiggy.outputs.Output):
    """
    Output messages to a file via MPI I/O.
    """

    def __init__(self, name, format, comm,
                 mode=MPI.MODE_CREATE | MPI.MODE_WRONLY,
                 close_atexit=True):
        self.filename = name
        self._format = format if format is not None else self._noop_format
        self.comm = comm
        self.mode = mode
        super(MPIOutput, self).__init__(format, close_atexit)

    def _open(self):
        self.file = MPI.File.Open(self.comm, self.filename,
                                  self.mode)

    def _close(self):
        self.file.Close()

    def _write(self, x):
        self.file.Iwrite_shared(x)

        # This seems to be necessary to prevent some log lines from being lost:
        self.file.Sync()
