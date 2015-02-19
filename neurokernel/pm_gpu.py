#!/usr/bin/env python

"""
Port mapper for GPU memory.
"""

import numbers

import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as elementwise
import pycuda.tools as tools

from plsel import PortMapper

class GPUPortMapper(PortMapper):
    """
    Maps a PyCUDA GPUArray to/from path-like port identifiers.
    """

    def _validate_data(self, data):
        """
        Check whether the mapper's ports are compatible with the specified port data array.
        """

        # None is valid because it is used to signify the absence of a data array:
        if data is None:
            return True
        try:
            # Can only handle 1D data arrays:
            assert np.ndim(data) == 1

            # The integers in the port map must be valid indices into the
            # data array:
            assert max(self.portmap) < len(data)

            # The port mapper may map identifiers to some portion of the data array:
            assert len(self) <= len(data)
        except:
            return False
        else:
            return True

    def __init__(self, selector, data=None, portmap=None, make_copy=True):
        super(PortMapper, self).__init__(selector, portmap)

        self._data = None
        if data is not None and make_copy:
            self.data = data.copy()
        else:
            self.data = data
    
    @property
    def data_ctype(self):
        """
        C type corresponding to type of data array.
        """
        
        if hasattr(self.data, 'dtype'):
            return tools.dtype_to_ctype(self.data.dtype)
        else:
            return ''

    @property
    def data(self):
        """
        Data associated with ports.
        """
        
        return self._data

    @data.setter
    def data(self, x):        
        if self._validate_data(x):
            if isinstance(x, gpuarray.GPUArray) or x is None:
                self._data = x
            else:
                self._data = gpuarray.to_gpu(x)
        else:
            raise ValueError('incompatible or invalid data array specified')

    def copy(self):
        """
        Return copy of this port mapper.

        Returns
        -------
        result : neurokernel.plsel.GPUPortMapper
            Copy of port mapper instance.
        """

        c = self.__class__('')
        c.portmap = self.portmap.copy()
        if self.data is not None:
            c.data = self.data.copy()
        return c

    @classmethod
    def from_pm(cls, pm):
        """
        Create a new port mapper instance given an existing instance.

        Parameters
        ----------
        pm : neurokernel.plsel.PortMapper
            Existing port mapper instance. If `pm` is not a GPUPortMapper, 

        Returns
        -------
        result : neurokernel.plsel.GPUPortMapper
            New port mapper instance.
        """

        assert isinstance(pm, PortMapper)
        r = cls('')
        r.portmap = pm.portmap.copy()
        if hasattr(pm, 'data') and pm.data is not None:
            r.data = pm.data.copy()
        return r

    def get_inds_nonzero(self):
        raise NotImplementedError

    def get_ports_nonzero(self):
        raise NotImplementedError

    def set(self, selector, data):
        self.set_by_inds(np.asarray(self.sel.select(self.portmap,
                        selector).dropna().values, dtype=np.int), data)

    def get(self, selector):
        return self.get_by_inds(np.asarray(self.sel.select(self.portmap, selector).dropna().values, dtype=np.int))

    __getitem__ = get
    __setitem__ = set

    def get_by_inds(self, inds):
        """
        Retrieve mapped data specified by integer index.
        
        Parameters
        ----------
        inds : sequence of int
            Integer indices of data elements to return.
        
        Returns
        -------
        result : numpy.ndarray
            Selected data.
        """

        assert len(np.shape(inds)) == 1
        assert issubclass(inds.dtype.type, numbers.Integral)

        N = len(inds)
        assert N <= len(self.data)

        if not isinstance(inds, gpuarray.GPUArray):
            inds = gpuarray.to_gpu(inds)
        result = gpuarray.empty(N, dtype=self.data.dtype)

        try:
            func = self.get_by_inds.cache[inds.dtype]
        except KeyError:
            inds_ctype = tools.dtype_to_ctype(inds.dtype)
            v = "{data_ctype} *dest, {inds_ctype} *inds, {data_ctype} *src".format(data_ctype=self.data_ctype, inds_ctype=inds_ctype)
            func = elementwise.ElementwiseKernel(v, "dest[i] = src[inds[i]]")
            self.get_by_inds.cache[inds.dtype] = func
        func(result, inds, self.data, range=slice(0, N, 1))
        return result.get()
    get_by_inds.cache = {}

    def set_by_inds(self, inds, data):
        """
        Set mapped data by integer indices.

        Parameters
        ----------
        inds : sequence of int
            Integer indices of data elements to update.
        data : numpy.ndarray
            Data to assign.
        """

        assert len(np.shape(inds)) == 1
        assert self.data.dtype == data.dtype
        assert issubclass(inds.dtype.type, numbers.Integral)

        N = len(inds)
        assert N == len(data)
        if not isinstance(inds, gpuarray.GPUArray):
            inds = gpuarray.to_gpu(inds)
        if not isinstance(data, gpuarray.GPUArray):
            data = gpuarray.to_gpu(data)
        try:
            func = self.set_by_inds.cache[inds.dtype]
        except KeyError:
            inds_ctype = tools.dtype_to_ctype(inds.dtype)
            v = "{data_ctype} *dest, {inds_ctype} *inds, {data_ctype} *src".format(data_ctype=self.data_ctype, inds_ctype=inds_ctype)        
            func = elementwise.ElementwiseKernel(v, "dest[inds[i]] = src[i]")
            self.set_by_inds.cache[inds.dtype] = func
        func(self.data, inds, data, range=slice(0, N, 1))
    set_by_inds.cache = {}
