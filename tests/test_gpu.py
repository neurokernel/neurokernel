#!/usr/bin/env python

from unittest import main, TestCase
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import neurokernel.tools.gpu as gpu

class test_gpu(TestCase):
    def test_get_by_inds(self):
        N = 100
        M = 10
        src = np.random.rand(N).astype(np.float32)
        ind = np.random.randint(0, N, M).astype(np.uint32)

        src_gpu = gpuarray.to_gpu(src)
        ind_gpu = gpuarray.to_gpu(ind)

        result = gpu.get_by_inds(src_gpu, ind_gpu)
        assert np.allclose(result, src[ind])

    def test_set_by_inds(self):
        # Set specified entries in dest_gpu to src_gpu:
        dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
        ind = gpuarray.to_gpu(np.array([0, 2, 4]))
        src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.float32))
        gpu.set_by_inds(dest_gpu, ind, src_gpu, 'dest')
        assert np.allclose(dest_gpu.get(), np.array([1, 1, 1, 3, 1], dtype=np.float32))

        # Set dest_gpu to specified entries in src_gpu:
        dest_gpu = gpuarray.to_gpu(np.zeros(3, dtype=np.float32))
        ind = gpuarray.to_gpu(np.array([0, 2, 4]))
        src_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
        gpu.set_by_inds(dest_gpu, ind, src_gpu, 'src')
        assert np.allclose(dest_gpu.get(), np.array([0, 2, 4], dtype=np.float32))

    def test_set_by_inds_from_inds(self):
        dest_gpu = gpuarray.to_gpu(np.zeros(5, dtype=np.float32))
        ind_dest = gpuarray.to_gpu(np.array([0, 2, 4]))
        src_gpu =  gpuarray.to_gpu(np.arange(5, 10, dtype=np.float32))
        ind_src =  gpuarray.to_gpu(np.array([2, 3, 4]))
        gpu.set_by_inds_from_inds(dest_gpu, ind_dest, src_gpu, ind_src)
        assert np.allclose(dest_gpu.get(), np.array([7, 0, 8, 0, 9], dtype=np.float32))

if __name__ == '__main__':
    main()

