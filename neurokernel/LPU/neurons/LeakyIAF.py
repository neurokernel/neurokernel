from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class LeakyIAF(BaseNeuron):
    def __init__(self, n_dict, spk, dt, debug=False):

        self.num = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = max( int(round) )
        self.debug = debug

        self.Vr  = garray.to_gpu( np.asarray( n_dict['Vr'], dtype=np.float64 ))
        self.Vt  = garray.to_gpu( np.asarray( n_dict['Vt'], dtype=np.float64 ))
        self.C   = garray.to_gpu( np.asarray( n_dict['C'], dtype=np.float64 ))
        self.R   = garray.to_gpu( np.asarray( n_dict['R'], dtype=np.float64 ))
        self.V   = garray.to_gpu( np.asarray( n_dcit['V0'], dtype=np.float64 ))
        self.spk = spk

        self.update = self.get_gpu_kernel()

    def eval( self, st = None):
        self.update.prepared_async_call(\
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num,\
            self.dt,\
            self.spk,\
            self.V.gpudata,\
            self.I.gpudata,\
            self.Vt.gpudata,\
            self.Vr.gpudata,\
            self.R.gpudata,\
            self.C.gpudata )

    def get_gpu_kernel( self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = ((self.num - 1) / self.gpu_block[0] + 1, 1)
        cuda_src = open( './leaky_iaf.cu','r')
        mod = SourceModule( \
                cuda_src.read() % {"type": dtype_to_ctype(np.float64),\
                                   "nneu": self.gpu_block[0] },\
                options=["--ptxas-options=-v"])
        func = mod.get_function("leaky_iaf")
        func.prepare( [ np.int32,   # neu_num
                        np.float64, # dt
                        np.intp,    # spk array
                        np.intp,    # V array
                        np.intp,    # I array
                        np.intp,    # Vt array
                        np.intp,    # Vr array
                        np.intp,    # R array
                        np.intp ])  # C array

        return func
