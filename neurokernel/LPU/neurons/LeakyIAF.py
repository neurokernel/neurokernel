from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_src = """
// %(type)s and %(nneu)d must be replaced using Python string foramtting
#define NNEU %(nneu)d

__global__ void leaky_iaf(
    int neu_num,
    %(type)s dt,
    int      *spk,
    %(type)s *V,
    %(type)s *I,
    %(type)s *Vt,
    %(type)s *Vr,
    %(type)s *R,
    %(type)s *C)
{
    int bid = blockIdx.x;
    int nid = bid * NNEU + threadIdx.x;

    %(type)s v,i,r,c;

    if( nid < neu_num ){
        v = V[nid];
        i = I[nid];
        r = R[nid];
        c = C[nid];

        // update v
        %(type)s bh = exp( -dt/r/c );
        v = v*bh + r*i*(1.0-bh);

        // spike detection
        spk[nid] = 0;
        if( v >= Vt[nid] ){
            v = Vr[nid];
            spk[nid] = 1;
        }

        V[nid] = v;
    }
    return;
}
"""

class LeakyIAF(BaseNeuron):
    def __init__(self, n_dict, spk, dt, debug=False):
        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = 1
        self.debug = debug

        self.Vr  = garray.to_gpu( np.asarray( n_dict['Vr'], dtype=np.float64 ))
        self.Vt  = garray.to_gpu( np.asarray( n_dict['Vt'], dtype=np.float64 ))
        self.C   = garray.to_gpu( np.asarray( n_dict['C'], dtype=np.float64 ))
        self.R   = garray.to_gpu( np.asarray( n_dict['R'], dtype=np.float64 ))
        self.V   = garray.to_gpu( np.asarray( n_dict['Vr'], dtype=np.float64 ))
        self.spk = spk

        self.update = self.get_gpu_kernel()

    @property
    def neuron_class(self): return True

    def eval( self, st = None):
        self.update.prepared_async_call(\
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num_neurons,\
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
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
        #cuda_src = open( './leaky_iaf.cu','r')
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64),\
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
