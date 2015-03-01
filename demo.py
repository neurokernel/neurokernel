#!/usr/bin/env python

"""
Sample program.
"""

import os
import sys

from mpi4py import MPI

from mpi_proc import Process, Manager

class MyProc(Process):
    def run(self):
        print '%s: %s, %s' % (self.rank, str(self._args), str(self._kwargs))
        if self.rank == 0:
            print '%s: %s' % (self.rank, self.recv_parent())

def myfunc(*args, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    print '%s: %s, %s' % (rank, str(args), str(kwargs))

if __name__ == '__main__':
    man = Manager()
    man.add(MyProc, 'x', 'y', z=1)
    man.add(MyProc, 'a', 'b', c=2)
    man.add(myfunc, 'p', 'q', r=3)
    man.run()
    man.send('xyz', 0)

