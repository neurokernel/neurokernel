#!/usr/bin/env python

"""
Sample program.
"""

import os
from StringIO import StringIO
import sys
from unittest import main, TestCase

from mpi4py import MPI

from neurokernel.mpi_proc import Process, ProcessManager

class MyProc(Process):
    def run(self):
        self.send_parent('%s %s %s' % (self.rank, str(self._args), 
                                       str(self._kwargs)))

def myfunc(*args, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    MPI.Comm.Get_parent().send('%s %s %s' % (rank, str(args), str(kwargs)), 0)

class test_mpi_proc(TestCase):
    def test_process(self):
        man = ProcessManager()
        man.add(MyProc, 'x', 'y', z=1)
        man.add(MyProc, 'a', 'b', c=2)
        man.run()
        results = []
        results.append(man.recv())
        results.append(man.recv())
        self.assertItemsEqual(results, 
                              ["0 ('x', 'y') {'z': 1}",
                               "1 ('a', 'b') {'c': 2}"])

    def test_func(self):
        man = ProcessManager()
        man.add(myfunc, 'x', 'y', z=1)
        man.add(myfunc, 'a', 'b', c=2)
        man.run()
        results = []
        results.append(man.recv())
        results.append(man.recv())
        self.assertItemsEqual(results, 
                              ["0 ('x', 'y') {'z': 1}",
                               "1 ('a', 'b') {'c': 2}"])
    
if __name__ == '__main__':
    main()
