#!/usr/bin/env python

import numbers

import numpy as np
import pycuda.driver as drv
import pycuda.elementwise as elementwise
import pycuda.gpuarray as gpuarray
from pycuda.tools import dtype_to_ctype

# List of available numerical types provided by numpy: 
# XXX This try/except is an ugly hack to prevent the doc build on
# ReadTheDocs from failing:
try:
    num_types = [np.typeDict[t] for t in \
                 np.typecodes['AllInteger']+np.typecodes['AllFloat']]
except TypeError:
    num_types = []

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

def bufint(a):
    """
    Return buffer interface to GPU or numpy array.

    Parameters
    ----------
    a : pycuda.gpuarray.GPUArray or numpy.ndarray
        GPU or numpy array.

    Returns
    -------
    b : buffer
        Buffer interface to array. Returns None if `a` has a length of 0.
    """

    if not a.size:
        return None
    else:
        if isinstance(a, gpuarray.GPUArray):
            return a.gpudata.as_buffer(a.nbytes)
        elif isinstance(a, np.ndarray):
            return a.data
        else:
            raise TypeError('argument must be a GPU or numpy array')

def get_by_inds(src_gpu, ind):
    """
    Get values in a GPUArray by index.

    Parameters
    ----------
    src_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance from which to get values.
    ind : pycuda.gpuarray.GPUArray or numpy.ndarray
        1D array of element indices to set. Must have an integer dtype.

    Returns
    -------
    result : numpy.ndarray
        Extracted data.

    Notes
    -----
    Only supports 1D index arrays.

    May not be efficient for certain index patterns because of lack of inability
    to coalesce memory operations.
    """

    dest_gpu = gpuarray.empty(ind.shape, src_gpu.dtype)
    set_by_inds(dest_gpu, ind, src_gpu, 'src')
    return dest_gpu.get()

def set_by_inds(dest_gpu, ind, src_gpu, ind_which='dest'):
    """
    Set values in a GPUArray by index.

    Parameters
    ----------
    dest_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance to modify.
    ind : pycuda.gpuarray.GPUArray or numpy.ndarray
        1D array of element indices to set. Must have an integer dtype.
    src_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance from which to set values.
    ind_which : str
        If set to 'dest', set the elements in `dest_gpu` with indices `ind`
        to the successive values in `src_gpu`; the lengths of `ind` and
        `src_gpu` must be equal. If set to 'src', set the
        successive values in `dest_gpu` to the values in `src_gpu` with indices
        `ind`; the lengths of `ind` and `dest_gpu` must be equal.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> from nk.tools.gpu import set_by_inds
    >>> dest_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
    >>> ind = gpuarray.to_gpu(np.array([0, 2, 4]))
    >>> src_gpu = gpuarray.to_gpu(np.array([1, 1, 1], dtype=np.float32))
    >>> set_by_inds(dest_gpu, ind, src_gpu, 'dest')
    >>> np.allclose(dest_gpu.get(), np.array([1, 1, 1, 3, 1], dtype=np.float32))
    True
    >>> dest_gpu = gpuarray.to_gpu(np.zeros(3, dtype=np.float32))
    >>> ind = gpuarray.to_gpu(np.array([0, 2, 4]))
    >>> src_gpu = gpuarray.to_gpu(np.arange(5, dtype=np.float32))
    >>> set_by_inds(dest_gpu, ind, src_gpu, 'src')
    >>> np.allclose(dest_gpu.get(), np.array([0, 2, 4], dtype=np.float32))
    True

    Notes
    -----
    Only supports 1D index arrays.

    May not be efficient for certain index patterns because of lack of inability
    to coalesce memory operations.
    """

    if np.isscalar(src_gpu) or np.isscalar(dest_gpu):
        raise ValueError('data must be array-like')
    if len(np.shape(ind)) > 1:
        raise ValueError('index array must be 1D')

    # Manually handle empty index array because it will cause the kernel to
    # fail if processed:
    N = len(ind)
    if N == 0:
        return

    if not issubclass(ind.dtype.type, numbers.Integral):
        raise ValueError('index array must contain integers')
    if not dest_gpu.dtype == src_gpu.dtype:
        raise ValueError('dest_gpu.dtype != src_gpu.dtype')

    if ind_which == 'dest':
        assert N == len(src_gpu)
    elif ind_which == 'src':
        assert N == len(dest_gpu)
    else:
        raise ValueError('invalid value for `ind_which`')
    if not isinstance(ind, gpuarray.GPUArray):
        ind = gpuarray.to_gpu(ind)
    try:
        func = set_by_inds.cache[(dest_gpu.dtype, ind.dtype, ind_which)]
    except KeyError:
        data_ctype = dtype_to_ctype(dest_gpu.dtype)
        ind_ctype = dtype_to_ctype(ind.dtype)        
        v = "{data_ctype} *dest, {ind_ctype} *ind, {data_ctype} *src".format(data_ctype=data_ctype, ind_ctype=ind_ctype)
    
        if ind_which == 'dest':
            func = elementwise.ElementwiseKernel(v, "dest[ind[i]] = src[i]")
        else:
            func = elementwise.ElementwiseKernel(v, "dest[i] = src[ind[i]]")
        set_by_inds.cache[(dest_gpu.dtype, ind.dtype, ind_which)] = func
    func(dest_gpu, ind, src_gpu, range=slice(0, N, 1))
set_by_inds.cache = {}

def set_by_inds_from_inds(dest_gpu, ind_dest, src_gpu, ind_src):
    """
    Set values in a GPUArray by index from indexed values in another GPUArray.

    Parameters
    ----------
    dest_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance to modify.
    ind_dest : pycuda.gpuarray.GPUArray or numpy.ndarray
        1D array of element indices in `dest_gpu` to set. Must have an integer dtype.
    src_gpu : pycuda.gpuarray.GPUArray
        GPUArray instance from which to set values.
    ind_src : pycuda.gpuarray.GPUArray or numpy.ndarray
        1D array of element indices in `src_gpu` to copy. Must have an integer
        dtype and be the same length as `ind_dest`.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> from nk.tools.gpu import set_by_inds_from_inds
    >>> dest_gpu = gpuarray.to_gpu(np.zeros(5, dtype=np.float32))
    >>> ind_dest = gpuarray.to_gpu(np.array([0, 2, 4]))
    >>> src_gpu =  gpuarray.to_gpu(np.arange(5, 10, dtype=np.float32))
    >>> ind_src =  gpuarray.to_gpu(np.array([2, 3, 4]))
    >>> gpu.set_by_inds_from_inds(dest_gpu, ind_dest, src_gpu, ind_src)
    >>> assert np.allclose(dest_gpu.get(), np.array([7, 0, 8, 0, 9], dtype=np.float32))
    True
    """

    if len(np.shape(ind_dest)) > 1:
        raise ValueError('destination index array must be 1D')
    if len(np.shape(ind_src)) > 1:
        raise ValueError('source index array must be 1D')
    assert dest_gpu.dtype == src_gpu.dtype
    assert ind_dest.dtype == ind_src.dtype
    assert issubclass(ind_dest.dtype.type, numbers.Integral)
    assert issubclass(ind_src.dtype.type, numbers.Integral)

    # Manually handle empty index array because it will cause the kernel to
    # fail if processed:
    N = len(ind_src)
    if N == 0:
        return
    assert N == len(ind_dest)
    if not isinstance(ind_dest, gpuarray.GPUArray):
        ind_dest = gpuarray.to_gpu(ind_dest)
    if not isinstance(ind_src, gpuarray.GPUArray):
        ind_src = gpuarray.to_gpu(ind_src)
    try:
        func = set_by_inds_from_inds.cache[(dest_gpu.dtype, ind_dest.dtype)]
    except KeyError:
        data_ctype = dtype_to_ctype(dest_gpu.dtype)
        ind_ctype = dtype_to_ctype(ind_dest.dtype)        
        v = "{data_ctype} *dest, {ind_ctype} *ind_dest,"\
            "{data_ctype} *src, {ind_ctype} *ind_src".format(data_ctype=data_ctype, ind_ctype=ind_ctype) 
        func = elementwise.ElementwiseKernel(v,
                "dest[ind_dest[i]] = src[ind_src[i]]")
        set_by_inds_from_inds.cache[(dest_gpu.dtype, ind_dest.dtype)] = func
    func(dest_gpu, ind_dest, src_gpu, ind_src, range=slice(0, N, 1))
set_by_inds_from_inds.cache = {}
