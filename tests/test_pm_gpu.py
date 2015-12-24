#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
from numpy.testing import assert_array_equal
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from pandas.util.testing import assert_series_equal

from neurokernel.pm_gpu import PortMapper, GPUPortMapper

class test_gpu_port_mapper(TestCase):
    def test_get(self):
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        res = pm['/foo[0:2]']
        assert_array_equal(data[0:2], res)

    def test_set_scalar(self):
        # Nonempty index array:
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        new_data = 1.0
        ind = np.array([0, 1])
        pm.set_by_inds(ind, new_data)
        assert_array_equal(np.full(ind.shape, new_data, type(new_data)),
                           pm.data.get()[0:2])

        # Empty index array:
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        new_data = 1.0
        pm.set_by_inds(np.array([], np.int_), new_data)
        assert_array_equal(data[0:2], pm.data.get()[0:2])

    def test_set_array(self):
        # Valid empty:
        pm = GPUPortMapper('/foo[0:3]')
        new_data = np.arange(2).astype(np.double)
        pm['/foo[0:2]'] = new_data
        assert_array_equal(new_data, pm.data.get()[0:2])

        # Valid nonempty:
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        new_data = np.arange(2).astype(np.double)
        pm['/foo[0:2]'] = new_data
        assert_array_equal(new_data, pm.data.get()[0:2])

    def test_get_by_inds(self):
        # Nonempty index array:
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        res = pm.get_by_inds(np.array([0, 1]))
        assert_array_equal(data[0:2], res)

        # Empty index array:
        res = pm.get_by_inds(np.array([], np.int_))
        assert len(res) == 0

    def test_set_by_inds_scalar(self):
        # Nonempty index array:
        pm = GPUPortMapper('/foo[0:3]', np.zeros(3, np.double))
        pm.set_by_inds(np.array([0, 1]), 1.0)
        assert_array_equal(np.ones(2, np.double), pm.data.get()[0:2])

        # Empty index array:
        pm = GPUPortMapper('/foo[0:3]', np.zeros(3, np.double))
        pm.set_by_inds(np.array([]), 1.0)
        assert_array_equal(np.zeros(2, np.double), pm.data.get()[0:2])

    def test_set_by_inds_array(self):
        # Nonempty index array:
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        new_data = np.arange(2).astype(np.double)
        pm.set_by_inds(np.array([0, 1]), new_data)
        assert_array_equal(new_data, pm.data.get()[0:2])

        # Empty index array:
        data = np.random.rand(3)
        pm = GPUPortMapper('/foo[0:3]', data)
        new_data = np.arange(2).astype(np.double)
        pm.set_by_inds(np.array([], np.int_), new_data)
        assert_array_equal(data[0:2], pm.data.get()[0:2])

    def test_from_pm_nongpu(self):
        # Empty:
        pm0 = PortMapper('/foo[0:3]')
        pm1 = GPUPortMapper.from_pm(pm0)
        assert_series_equal(pm0.portmap, pm1.portmap)

        # Nonempty:
        data = np.random.rand(3)
        pm0 = PortMapper('/foo[0:3]', data)
        pm1 = GPUPortMapper.from_pm(pm0)
        assert_array_equal(pm0.data, pm1.data.get())
        assert_series_equal(pm0.portmap, pm1.portmap)

    def test_from_pm_gpu(self):
        # Empty:
        pm0 = GPUPortMapper('/foo[0:3]')
        pm1 = GPUPortMapper.from_pm(pm0)
        assert_series_equal(pm0.portmap, pm1.portmap)

        # Nonempty:
        data = np.random.rand(3)
        pm0 = GPUPortMapper('/foo[0:3]', data)
        pm1 = GPUPortMapper.from_pm(pm0)
        assert_array_equal(pm0.data.get(), pm1.data.get())
        assert_series_equal(pm0.portmap, pm1.portmap)
        assert pm0.data.ptr != pm1.data.ptr

    def test_copy(self):
        # Empty:
        pm0 = GPUPortMapper('/foo[0:5]')
        pm1 = pm0.copy()
        assert_series_equal(pm0.portmap, pm1.portmap)
        assert pm0.data is None and pm1.data is None

        # Nonempty:
        data = np.random.rand(5)
        pm0 = GPUPortMapper('/foo[0:5]', data)
        pm1 = pm0.copy()
        assert_array_equal(pm0.data.get(), pm1.data.get())
        assert_series_equal(pm0.portmap, pm1.portmap)
        assert pm0.data.ptr != pm1.data.ptr

if __name__ == '__main__':
    main()
