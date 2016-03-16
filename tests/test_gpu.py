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
        np.allclose(result, src[ind])

if __name__ == '__main__':
    main()

