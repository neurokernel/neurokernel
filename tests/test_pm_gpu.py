#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
from numpy.testing import assert_array_equal
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from neurokernel.pm_gpu import GPUPortMapper

class test_gpu_port_mapper(TestCase):
    def test_get_by_inds(self):
        data = np.random.rand(3)
        data_gpu = gpuarray.to_gpu(data)
        pm = GPUPortMapper('/foo[0:3]', data_gpu)
        res_gpu = pm.get_by_inds(np.array([0, 1]))
        assert_array_equal(data[0:2], res_gpu.get())

    def test_set_by_inds(self):
        data = np.random.rand(3)
        data_gpu = gpuarray.to_gpu(data)
        pm = GPUPortMapper('/foo[0:3]', data_gpu)
        new_data = np.arange(2).astype(np.double)
        new_data_gpu = gpuarray.to_gpu(new_data)
        pm.set_by_inds(np.array([0, 1]), new_data_gpu)
        assert_array_equal(new_data, pm.data.get()[0:2])

if __name__ == '__main__':
    main()
