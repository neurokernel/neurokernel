#!/usr/bin/env python

"""
Sample program.
"""

import os
from io import BytesIO
from io import StringIO
import sys
from unittest import main, TestCase

from mpi4py import MPI

from neurokernel.mpi_proc import Process, ProcessManager

import logging

class MyProc(Process):
    def __init__(self, a, b, c, routing_table=None):
        super(MyProc, self).__init__()
        self.response = "a=%s b=%s c=%s" % (a,b,c)

    def run(self):
        self.send_parent(self.response)

class test_mpi_proc(TestCase):
    def test_process(self):
        man = ProcessManager()
        man.add(MyProc, 'x', 'y', c=1)
        man.spawn()
        results=man.recv()
        self.assertEqual(results,'a=x b=y c=1')

if __name__ == '__main__':
    main()
