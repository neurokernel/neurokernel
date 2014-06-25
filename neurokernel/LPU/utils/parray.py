#!/usr/bin/env python

#This is the parray class (pitched array) that serves as a complement to pycuda.GPUArray.
#The intention here is to automatically create 2D or 3D array with a pitched structure
#and varies operation on it.

import pycuda.driver as cuda
import numpy as np
from pytools import memoize
import parray_utils as pu

""" utilities"""
@memoize
def _splay_backend(n, M):
    
    block_count = 6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT
    
    if M <= 1:
        block = (256, 1, 1)
    else:
        block = (32, 8, 1)
    
    return (block_count, 1), block


def splay(n, M):
    return _splay_backend(n, M)
    


def _get_common_dtype(obj1, obj2):
    return (obj1.dtype.type(0) + obj2.dtype.type(0)).dtype
    
def _pd(shape):
    s = 1
    for dim in shape[1:]:
        s *= dim
    return s

def _assignshape(shape, axis, value):
    a = []
    for i in range(len(shape)):
        if i == axis:
            a.append(value)
        else:
            a.append(shape[i])
    return tuple(a) 


def PitchTrans(shape, dst, dst_ld, src, src_ld, dtype, aligned=False, async = False, stream = None):    
    size = np.dtype(dtype).itemsize
    
    
    trans = cuda.Memcpy2D()
    trans.src_pitch = src_ld * size
    if isinstance(src, (cuda.DeviceAllocation, int, long)):
        trans.set_src_device(src)
    else:
        trans.set_src_host(src)
    
    trans.dst_pitch = dst_ld * size
    if isinstance(dst, (cuda.DeviceAllocation, int, long)):
        trans.set_dst_device(dst)
    else:
        trans.set_dst_host(dst)
    
    trans.width_in_bytes = _pd(shape) * size
    trans.height = int(shape[0])
    
    if async:
        trans(stream)
    else:
        trans(aligned = aligned)
        
"""end of utilities"""  




class PitchArray(object):
    def __init__(self, shape, dtype, gpudata=None, pitch = None):
        """create a PitchArray
        shape: shape of the array
        dtype: dtype of the array
        gpudata: DeviceAllocation object indicating the device memory allocated
        pitch: if gpudata is specified and pitch is True, gpudata will be treated
                as if it was allocated by cudaMallocPitch with pitch

        attributes:
        .shape: shape of self
        .size:  number of elements of the array
        .mem_size: number of elements of total memory allocated
        .ld: leading dimension
        .M: 1 if self is a vector, shape[0] otherwise
        .N: self.size if self is a vector, product of shape[1] and shape[2] otherwise
        .gpudata: DeviceAllocation
        .ndim: number of dimensions
        .dtype: dtype of array
        self.nbytes: total memory allocated for the array in bytes
        
        Note:
        any 1-dim shape will result in a row vector with new shape as (1, shape)

        operations of PitchArray is elementwise operation

        """
    
        try:
            tmpshape = []
            s = 1
            for dim in shape:
                dim = int(dim)
                assert isinstance(dim, int)
                s *= dim
                tmpshape.append(dim)
                
            self.shape = tuple(tmpshape)
        except TypeError:
            s = int(shape)
            assert isinstance(s, int)
            if s:
                self.shape = (1, s)
            else:
                self.shape = (0, 0)
            
        self.ndim = len(self.shape)
        
        if self.ndim > 3:
            raise ValueError("Only support array of dimension leq 3")
        
        self.dtype = np.dtype(dtype)
        
        self.size = s
        
        
        if gpudata is None:
            if self.size:
                if _pd(self.shape) == 1 or self.shape[0] == 1:
                    self.gpudata = cuda.mem_alloc(self.size * self.dtype.itemsize)
                    self.mem_size = self.size
                    self.ld = _pd(self.shape)
                    self.M = 1
                    self.N = self.size
                    
                else:
                    self.gpudata, pitch = cuda.mem_alloc_pitch(int(_pd(self.shape) * np.dtype(dtype).itemsize), self.shape[0], np.dtype(dtype).itemsize)
                    self.ld = pitch / np.dtype(dtype).itemsize
                    self.mem_size = self.ld * self.shape[0]
                    self.M = self.shape[0]
                    self.N = _pd(self.shape)
            
            else:
                self.gpudata = None
                self.M = 0
                self.N = 0
                self.ld = 0
                self.mem_size = 0
                
        else:
            #assumed that the device memory was also allocated by mem_alloc_pitch is required by the shape
            assert gpudata.__class__ == cuda.DeviceAllocation
            
            if self.size:
                self.gpudata = gpudata
                if _pd(self.shape) == 1 or self.shape[0] == 1:
                    self.mem_size = self.size
                    self.ld = _pd(self.shape)
                    self.M = 1
                    self.N = self.size
                else:
                    if pitch is None:
                        pitch = int(np.ceil(float(_pd(self.shape) * np.dtype(dtype).itemsize) / 512) * 512)
                    else:
                        assert pitch == int(np.ceil(float(_pd(self.shape) * np.dtype(dtype).itemsize) / 512) * 512)
                    
                    self.ld = pitch / np.dtype(dtype).itemsize
                    self.mem_size = self.ld * self.shape[0]
                    self.M = self.shape[0]
                    self.N = _pd(self.shape)
                        
            else:
                self.gpudata = None
                self.M = 0
                self.N = 0
                self.ld = 0
                self.mem_size = 0
                print "warning: shape may not be assigned properly"
        self.nbytes = self.dtype.itemsize * self.mem_size
        self._grid, self._block = splay(self.mem_size, self.M)
    
        
    def set(self, ary):
        """
        set PitchArray with an ndarray
        """
        assert ary.ndim <= 3
        assert ary.dtype == ary.dtype
        
        assert ary.size == self.size
        
        if self.size:
            if self.M == 1:
                cuda.memcpy_htod(self.gpudata, ary)
            else:
                PitchTrans(self.shape, self.gpudata, self.ld, ary, _pd(self.shape), self.dtype)
    
    def set_async(self, ary, stream=None):
        assert ary.ndim <= 3
        assert ary.dtype == ary.dtype
        
        assert ary.size == self.size
        
        if ary.base.__class__ != cuda.HostAllocation:
                raise TypeError("asynchronous memory trasfer requires pagelocked numpy array")
                
        if self.size:
            if self.M == 1:
                cuda.memcpy_htod_async(self.gpudata, ary, stream)
            else:
                PitchTrans(self.shape, self.gpudata, self.ld, ary, _pd(self.shape), self.dtype, async = True, stream = stream)


    def get(self, ary = None, pagelocked = False):
        """
        get the PitchArray to an ndarray
        if ary is specified, will transfer device memory to ary's memory
        pagelocked is ary's memory is pagelocked
        """
        if ary is None:
            if pagelocked:
                ary = cuda.pagelocked_empty(self.shape, self.dtype)
            else:
                ary = np.empty(self.shape, self.dtype)
        else:
            assert ary.size == self.size
            assert ary.dtype == ary.dtype

        
        if self.size:
            if self.M == 1:
                cuda.memcpy_dtoh(ary, self.gpudata)
            else:
                PitchTrans(self.shape, ary, _pd(self.shape), self.gpudata, self.ld, self.dtype)
                
        return ary


    def get_async(self, stream = None, ary = None):
        if ary is None:
            ary = cuda.pagelocked_empty(self.shape, self.dtype)
            
        else:
            assert ary.size == self.size
            assert ary.dtype == ary.dtype
            if ary.base.__class__ != cuda.HostAllocation:
                raise TypeError("asynchronous memory trasfer requires pagelocked numpy array")
        
        if self.size:
            if self.M == 1:
                cuda.memcpy_dtoh_async(ary, self.gpudata, stream)
            else:
                PitchTrans(self.shape, ary, _pd(self.shape), self.gpudata, self.ld, self.dtype, async = True, stream = stream)
                
        return ary

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

    def __hash__(self):
        raise TypeError("PitchArrays are not hashable.")
    
    def _new_like_me(self, dtype = None):
        if dtype is None:
            dtype = self.dtype
        return self.__class__(self.shape, dtype)
    
    def __add__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_addarray_function(self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_addarray_function(self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        else:
            if other == 0:
                return self.copy()
            else:
                result = self._new_like_me()
                if self.size:
                    if self.M == 1:
                        func = pu.get_addscalar_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_addscalar_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other)
                return result

    __radd__ = __add__
    
    def __sub__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        else:
            if other == 0:
                return self.copy()
            else:
                result = self._new_like_me()
                if self.size:
                    if self.M == 1:
                        func = pu.get_subscalar_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_subscalar_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other)
                return result

    def __rsub__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(other.dtype, self.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, other.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(other.dtype, self.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, other.gpudata, other.ld, self.gpudata, self.ld)
            return result
        else:
        
            result = self._new_like_me()
            if self.size:
                if self.M == 1:
                    func = pu.get_scalarsub_function(self.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other, self.size)
                else:
                    func = pu.get_scalarsub_function(self.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other)
            return result


    def __mul__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_mularray_function(self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_mularray_function(self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        else:
            if other == 1.0:
                return self.copy()
            else:
                result = self._new_like_me()
                if self.size:
                    if self.M == 1:
                        func = pu.get_mulscalar_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_mulscalar_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other)
                return result

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        else:
            if other == 1.0:
                return self.copy()
            else:
                result = self._new_like_me()
                if self.size:
                    if self.M == 1:
                        func = pu.get_divscalar_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_divscalar_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other)
                return result

    def __rdiv__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            result = self._new_like_me(_get_common_dtype(self, other))
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(other.dtype, self.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, other.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(other.dtype, self.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, other.gpudata, other.ld, self.gpudata, self.ld)
            return result
        else:
            if other == 1.0:
                return self.copy()
            else:
                result = self._new_like_me()
                if self.size:
                    if self.M == 1:
                        func = pu.get_scalardiv_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_scalardiv_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other)
                return result

    def __neg__(self):
        return 0-self

    def __iadd__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_addarray_function(self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_addarray_function(self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        else:
            if other == 0:
                return self
            else:
                if self.size:
                    if self.M == 1:
                        func = pu.get_addscalar_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, self.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_addscalar_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, self.gpudata, self.ld, self.gpudata, self.ld, other)
                return self

    def __isub__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        else:
            if other == 0:
                return self
            else:
                if self.size:
                    if self.M == 1:
                        func = pu.get_subscalar_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, self.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_subscalar_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, self.gpudata, self.ld, self.gpudata, self.ld, other)
                return self

    def __imul__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_mularray_function(self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_mularray_function(self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        else:
            if other == 1.0:
                return self
            else:
                if self.size:
                    if self.M == 1:
                        func = pu.get_mulscalar_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, self.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_mulscalar_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, self.gpudata, self.ld, self.gpudata, self.ld, other)
                return self

    def __idiv__(self, other):
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(self.dtype, other.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, other.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(self.dtype, other.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld, other.gpudata, other.ld)
            return result
        else:
            if other == 1.0:
                return self
            else:
                if self.size:
                    if self.M == 1:
                        func = pu.get_divscalar_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, self.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_divscalar_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, self.gpudata, self.ld, self.gpudata, self.ld, other)
                return self


    def add(self, other):
        """
        add other to self
        inplace if possible
        """
        return self.__iadd__(other)

    def sub(self, other):
        """
        substract other from self
        inplace if possible
        """
        return self.__isub__(other)

    def rsub(self, other):
        """
        substract other by self
        inplace if possible
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_subarray_function(other.dtype, self.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, other.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_subarray_function(other.dtype, self.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, other.gpudata, other.ld, self.gpudata, self.ld)
            return result
        else:
            if other == 0:
                return self
            else:
                if self.size:
                    if self.M == 1:
                        func = pu.get_scalarsub_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, self.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_scalarsub_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, self.gpudata, self.ld, self.gpudata, self.ld, other)
                return self

    def mul(self, other):
        """
        multiply other with self
        inplace if possible
        """
        return self.__imul__(other)

    def div(self, other):
        """
        divide other from self
        inplace if possible
        """
        return self.__idiv__(other)

    def rdiv(self, other):
        """
        divide other by self
        inplace if possible
        """
        if isinstance(other, PitchArray):
            if self.shape != other.shape:
                raise ValueError("array dimension misaligned")
            dtype = _get_common_dtype(self, other)
            if self.dtype == dtype:
                result = self
            else:
                result = self._new_like_me(dtype = dtype)
                
            if self.size:
                if self.M == 1:
                    func = pu.get_divarray_function(other.dtype, self.dtype, result.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, other.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_divarray_function(other.dtype, self.dtype, result.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, other.gpudata, other.ld, self.gpudata, self.ld)
            return result
        else:
            if other == 0:
                return self
            else:
                if self.size:
                    if self.M == 1:
                        func = pu.get_scalardiv_function(self.dtype, pitch = False)
                        func.prepared_call(self._grid, self._block, self.gpudata, self.gpudata, other, self.size)
                    else:
                        func = pu.get_scalardiv_function(self.dtype, pitch = True)
                        func.prepared_call(self._grid, self._block, self.M, self.N, self.gpudata, self.ld, self.gpudata, self.ld, other)
                return self

    def fill(self, value, stream=None):
        """
        fill all entries of self with value

        """
        if self.size:
            if self.M == 1:
                func = pu.get_fill_function(self.dtype, pitch = False)
                func.prepared_call(self._grid, self._block, self.size, self.gpudata, value)
            else:
                func = pu.get_fill_function(self.dtype, pitch = True)
                
                func.prepared_call(self._grid, self._block, self.M, self.N, self.gpudata, self.ld, value)
    
    def copy(self):
        """
        returns a duplicated copy of self
        """
        result = self._new_like_me()
        if self.size:
            cuda.memcpy_dtod(result.gpudata, self.gpudata, self.mem_size * self.dtype.itemsize)
        
        return result
    
    
    def real(self):
        """
        returns the real part of self
        """
        if self.dtype == np.complex128:
            result = self._new_like_me(dtype = np.float64)
        elif self.dtype == np.complex64:
            result = self._new_like_me(dtype = np.float32)
        else:
            return self
        
        if self.size:
            if self.M == 1:
                func = pu.get_realimag_function(self.dtype, real = True, pitch = False)
                func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, self.size)
            else:
                func = pu.get_realimag_function(self.dtype, real = True, pitch = True)
                func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld)
        return result
    
    def imag(self):
        """
        returns the imaginary part of self
        """
        if self.dtype == np.complex128:
            result = self._new_like_me(dtype = np.float64)
        elif self.dtype == np.complex64:
            result = self._new_like_me(dtype = np.float32)
        else:
            return zeros_like(self)
        
        if self.size:
            if self.M == 1:
                func = pu.get_realimag_function(self.dtype, real = False, pitch = False)
                func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, self.size)
            else:
                func = pu.get_realimag_function(self.dtype, real = False, pitch = True)
                func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld)
        return result
        
    def abs(self):
        """
        returns the absolute value of self
        """
        if self.dtype in [np.complex128, np.float64]:
            result = self._new_like_me(dtype = np.float64)
        elif self.dtype in [np.complex64, np.float32]:
            result = self._new_like_me(dtype = np.float32)
        else:
            result = self._new_like_me()
            
        if self.M == 1:
            func = pu.get_abs_function(self.dtype, pitch = False)
            func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, self.size)
        else:
            func = pu.get_abs_function(self.dtype, pitch = True)
            func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld)
        return result
    
    def conj(self, inplace = True):
        """
        returns the conjuation of self.
        if inplace is True, conjugation will be performed in place

        """
        if self.dtype in [np.complex64, np.complex128]:
            if inplace:
                result = self
            else:
                result = self._new_like_me()
            
            if self.M == 1:
                func = pu.get_conj_function(self.dtype, pitch = False)
                func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, self.size)
            else:
                func = pu.get_conj_function(self.dtype, pitch = True)
                func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld)
        else:
            result = self

        return result
    
    def reshape(self, shape, inplace = False):
        """
        reshape the shape of self to "shape"
        if inplace is Ture, enforce to keep the device memory of self if possible
        """
        
        sx = 1
        for dim in self.shape:
            sx *= dim
        
        s = 1
        flag = False
        n = 0
        axis = 0
        idx = -1
        for dim in shape:
            if dim == -1:
                flag = True
                n += 1
                idx = axis
            else:
                s *= dim
            axis += 1
        
        if flag:
            if n > 1:
                raise ValueError("can only specify one unknown dimension")
            else:
                if sx % s == 0:
                    shape = _assignshape(shape, idx, int(sx / s))
                else:
                    raise ValueError("cannot infer the size of the remaining axis")
        else:
            if s != sx:
                raise ValueError("total size of new array must be unchanged")

        if inplace:
            if shape[0] == self.shape[0]:
                self.shape = shape
                return self
            else:
                raise ValueError("cannot resize inplacely")

        
        result = PitchArray(shape, self.dtype)
        func = pu.get_resize_function(self.dtype)
        #func.set_block_shape(256,1,1)
        func.prepared_call(self._grid, (256,1,1), self.shape[0], _pd(self.shape), result.shape[0], _pd(result.shape), result.gpudata, result.ld, self.gpudata, self.ld)
        return result
    
    
    
    def astype(self, dtype):
        """ convert dtype of self to dtype """
        if self.dtype == dtype:
            return self.copy()
        else:
            result = self._new_like_me(dtype = dtype)
            
            if self.size:
                if self.M == 1:
                    func = pu.get_astype_function(dtype, self.dtype, pitch = False)
                    func.prepared_call(self._grid, self._block, result.gpudata, self.gpudata, self.size)
                else:
                    func = pu.get_astype_function(dtype, self.dtype, pitch = True)
                    func.prepared_call(self._grid, self._block, self.M, self.N, result.gpudata, result.ld, self.gpudata, self.ld)
            return result
    
    def T(self):
        """returns the transpose, PitchArray must be 2 dimensional"""
        
        if len(self.shape) > 2:
            raise ValueError("transpose only apply to 2D matrix")
        
        shape_t = self.shape[::-1]
        
        if self.M == 1:
            result = self.copy()
            result.shape = shape_t
            result.ld = _pd(result.shape)
            return result

        result = PitchArray(shape_t, self.dtype)

        if self.size:
            func = pu.get_transpose_function(self.dtype)
            func.prepared_call(self._grid, self._block, self.shape[0], self.shape[1], result.gpudata, result.ld, self.gpudata, self.ld)
        
        return result
    
    def H(self):
        """returns the conjugate transpose, PitchArray must be 2 dimensional"""
        if len(self.shape) > 2:
            raise ValueError("transpose only apply to 2D matrix")
        
        shape_t = self.shape[::-1]
        
        
        if self.M == 1:
            result = conj(self)
            result.shape = shape_t
            result.ld = _pd(result.shape)
            return result

        result = PitchArray(shape_t, self.dtype)

        if self.size:
            func = pu.get_transpose_function(self.dtype, conj = True)
            func.prepared_call(self._grid, self._block, self.shape[0], self.shape[1], result.gpudata, result.ld, self.gpudata, self.ld)
        
        return result
    
    
    
    def copy_rows(self, start, stop, step = 1):
        nrows = len(range(start,stop,step))
        if nrows:
            
            if self.ndim == 2:
                shape = (nrows, self.shape[1])
            else:
                shape = (nrows, self.shape[1], self.shape[2])
        else:
            if self.ndim == 2:
                shape = (nrows, 0)
            else:
                shape = (nrows, 0, 0)
        
        result = PitchArray(shape, self.dtype)
        
        if nrows > 1:
            PitchTrans(shape, result.gpudata, result.ld, int(self.gpudata) + start * self.ld * self.dtype.itemsize, self.ld * step, self.dtype)
        elif nrows == 1:
            cuda.memcpy_dtod(result.gpudata, int(self.gpudata) + start * self.ld * self.dtype.itemsize, self.dtype.itemsize * _pd(shape))
        return result
        
        

def to_gpu(ary):
    """ transfer a numpy ndarray to a PitchArray """
    result = PitchArray(ary.shape, ary.dtype)
    result.set(ary)
    return result


def to_gpu_async(ary, stream = None):
    result = PitchArray(ary.shape, ary.dtype)
    result.set_async(ary, stream)


empty = PitchArray

def empty_like(other_ary):
    result = PitchArray(other_ary.shape, other_ary.dtype)
    return result


def zeros(shape, dtype):
    result = PitchArray(shape, dtype)
    result.fill(0)
    return result

def zeros_like(other_ary):
    result = PitchArray(other_ary.shape, other_ary.dtype)
    result.fill(0)
    return result


def ones(shape, dtype):
    result = PitchArray(shape, dtype)
    result.fill(1)
    return result

def ones_like(other_ary):
    result = PitchArray(other_ary.shape, other_ary.dtype)
    result.fill(1)
    return result    

    
def make_pitcharray(dptr, shape, dtype, linear = False, pitch=None):
    """
    create a PitchArray from a DeviceAllocation pointer
    linear: "True" indicates the device memory is a linearly allocated 
            "False" indicates the device memory is allocated by cudaMallocPitch,
            and pitch must be provided
    """
    
    if linear:
        result = PitchArray(shape, dtype)
        if result.size:
            if result.M == 1:
                cuda.memcpy_dtod(result.gpudata, dptr, result.mem_size * dtype.itemsize)
            else:
                PitchTrans(shape, result.gpudata, result.ld, dptr, _pd(shape), dtype)
                
    else:
        result = PitchArray(shape, dtype, gpudata=dptr, pitch = pitch)
    
    return result


def arrayg2p(other_gpuarray):
    """convert a GPUArray to a PitchArray"""
    result = make_pitcharray(other_gpuarray.gpudata, other_gpuarray.shape, other_gpuarray.dtype, linear = True)
    return result



def arrayp2g(pary):
    """convert a PitchArray to a GPUArray"""
    from pycuda.gpuarray import GPUArray
    result = GPUArray(pary.shape, pary.dtype)
    if pary.size:
        if pary.M == 1:
            cuda.memcpy_dtod(result.gpudata, pary.gpudata, pary.mem_size * pary.dtype.itemsize)
        else:
            PitchTrans(pary.shape, result.gpudata, _pd(result.shape), pary.gpudata, pary.ld, pary.dtype)
            
    return result


def conj(pary):
    """ returns the conjugation of 2D PitchArray"""
    return pary.conj(inplace = False)







