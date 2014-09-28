#!/usr/bin/env python

import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pytools

# List of available numerical types provided by numpy: 
num_types = [np.typeDict[t] for t in \
             np.typecodes['AllInteger']+np.typecodes['AllFloat']]

# Numbers of bytes occupied by each numerical type:
num_nbytes = dict((np.dtype(t), t(1).nbytes) for t in num_types)

def set_realloc(x_gpu, data):
    """
    Transfer data into a GPUArray instance.

    Copies the contents of a numpy array into a GPUArray instance. If
    the array has a different type or dimensions than the instance,
    the GPU memory used by the instance is reallocated and the
    instance updated appropriately.
    
    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance to modify.
    data : numpy.ndarray
        Array of data to transfer to the GPU.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import misc
    >>> x = np.asarray(np.random.rand(5), np.float32)
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> x = np.asarray(np.random.rand(10, 1), np.float64)
    >>> set_realloc(x_gpu, x)
    >>> np.allclose(x, x_gpu.get())
    True
    
    """

    # Only reallocate if absolutely necessary:
    if x_gpu.shape != data.shape or x_gpu.size != data.size or \
        x_gpu.strides != data.strides or x_gpu.dtype != data.dtype:
        
        # Free old memory:
        x_gpu.gpudata.free()

        # Allocate new memory:
        nbytes = num_nbytes[data.dtype]
        x_gpu.gpudata = drv.mem_alloc(nbytes*data.size)
    
        # Set array attributes:
        x_gpu.shape = data.shape
        x_gpu.size = data.size
        x_gpu.strides = data.strides
        x_gpu.dtype = data.dtype
        
    # Update the GPU memory:
    x_gpu.set(data)

@pytools.memoize
def _get_extract_kernel():
    return SourceModule("""
    __global__ void func(double *x, double *y, unsigned int *idx, unsigned int M) {
        unsigned int tid = threadIdx.x;
        unsigned int total_threads = gridDim.x*blockDim.x;
        unsigned int block_start = blockDim.x*blockIdx.x;
        unsigned int i;

        for (i = block_start+tid; i < M; i += total_threads)
            y[i] = x[idx[i]];
      }
    """)

@pytools.memoize
def _get_unextract_kernel():
    return SourceModule("""
    __global__ void func(double *x, double *y, unsigned int *idx, unsigned int M) {
        unsigned int tid = threadIdx.x;
        unsigned int total_threads = gridDim.x*blockDim.x;
        unsigned int block_start = blockDim.x*blockIdx.x;
        unsigned int i;

        for (i = block_start+tid; i < M; i += total_threads)
            y[idx[i]] = x[i];
       }
    """)

def extract_contiguous(from_gpu, to_gpu, idx_gpu, dev):
    """
    Copy select discontiguous elements to a contiguous array.

    Parameters
    ----------
    from_gpu : pycuda.GPUArray
        Source array.
    to_gpu : pycuda.GPUArray
        Destination array.
    idx_gpu : pycuda.GPUArray
        Indices of elements from `from_gpu` to copy to `to_gpu`.
    dev : pycuda.driver.Device
        GPU device hosting `from_gpu` and `to_gpu`.
    """

    M = np.uint32(len(idx_gpu))
    grid, block = gpuarray.splay(M, dev)
    func = _get_extract_kernel().get_function('func')
    func(from_gpu, to_gpu, idx_gpu, M, block=block, grid=grid)

def unextract_contiguous(from_gpu, to_gpu, idx_gpu, dev):
    """
    Copy contiguous elements to a select elements in a discontiguous array.

    Parameters
    ----------
    from_gpu : pycuda.GPUArray
        Source array.
    to_gpu : pycuda.GPUArray
        Destination array.
    idx_gpu : pycuda.GPUArray
        Indices of elements in `to_gpu` to which the contents of 
        `from_gpu` are to be copied.
    dev : pycuda.driver.Device
        GPU device hosting `from_gpu` and `to_gpu`.
    """

    M = np.uint32(len(idx_gpu))
    grid, block = gpuarray.splay(M, dev)
    func = _get_unextract_kernel().get_function('func')
    func(from_gpu, to_gpu, idx_gpu, M, block=block, grid=grid)
