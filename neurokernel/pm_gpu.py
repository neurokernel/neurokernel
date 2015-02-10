#!/usr/bin/env python

"""
Port mapper for GPU memory.
"""

import numpy as np
import pycuda.gpuarray as gpuarray

from plsel import BasePortMapper

class GPUPortMapper(PortMapper):
    def __init__(self, selector, data=None, portmap=None, make_copy=True):
        super(PortMapper, self).__init__(selector, portmap)
        N = len(self)

        if data is None or len(data) == 0:
            self.data = gpuarray.empty(0, np.double)
        else:
            assert np.ndim(data) == 1
            assert type(data) == gpuarray.GPUArray
            
            # The integers in the port map must be valid indices into the
            # data array:
            assert max(self.portmap) < len(data)

            # The port mapper may map identifiers to some portion of the data array:
            assert N <= len(data)
            if make_copy:
                self.data = data.copy()
            else:
                self.data = data

    def get_inds_nonzero(self):
        raise NotImplementedError

    def get_ports_nonzero(self):
        raise NotImplementedError

    def get_by_inds(self, inds):
        raise NotImplementedError

    def set_by_ind(self, inds, data):
        # Can be implemented using scikits.cuda.misc.set_by_index
        raise NotImplementedError


