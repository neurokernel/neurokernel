#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
from numpy.testing import assert_array_equal
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from neurokernel.pm_gpu import GPUPortMapper

class test_gpu_port_mapper(TestCase):
    def test_get(self):
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        res = pm['/foo[0:2]']
        assert_array_equal(data[0:2], res)

    def test_set(self):
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        new_data = np.arange(2).astype(np.double)
        pm['/foo[0:2]'] = new_data
        assert_array_equal(new_data, pm.data.get()[0:2])

    def test_get_by_inds(self):
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        res = pm.get_by_inds(np.array([0, 1]))
        assert_array_equal(data[0:2], res)

    def test_set_by_inds(self):
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        new_data = np.arange(2).astype(np.double)
        pm.set_by_inds(np.array([0, 1]), new_data)
        assert_array_equal(new_data, pm.data.get()[0:2])

    def test_from_pm(self):
        data = np.random.rand(3)
        pm0 = GPUPortMapper('/foo[0:3]', data)
        pm1 = GPUPortMapper.from_pm(pm0)
        assert_array_equal(pm0.data.get(), pm1.data.get())
        assert pm0.data.ptr != pm1.data.ptr

if __name__ == '__main__':
    main()
