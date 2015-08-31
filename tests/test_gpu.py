#!/usr/bin/env python

from unittest import main, TestCase
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import neurokernel.tools.gpu as gpu

class test_gpu(TestCase):
    def test_extract_contiguous(self):
        N = 100
        M = 10
        x = np.random.rand(N).astype(np.float32)
        y = np.empty(M, np.float32)
        idx = np.random.randint(0, N, M).astype(np.uint32)

        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        idx_gpu = gpuarray.to_gpu(idx)

        gpu.extract_contiguous(x_gpu, y_gpu, idx_gpu, pycuda.autoinit.device)
        np.allclose(y_gpu.get(), x[idx])

    def test_unextract_contiguous(self):
        N = 100
        M = 10
        x = np.random.rand(M).astype(np.float32)
        y = np.empty(N, np.float32)
        idx = np.random.randint(0, N, M).astype(np.uint32)

        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.to_gpu(y)
        idx_gpu = gpuarray.to_gpu(idx)

        gpu.unextract_contiguous(x_gpu, y_gpu, idx_gpu, pycuda.autoinit.device)
        np.allclose(y_gpu.get()[idx], x)

if __name__ == '__main__':
    main()

