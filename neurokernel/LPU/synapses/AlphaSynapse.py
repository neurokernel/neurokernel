from basesynapse import BaseSynapse

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_src = """
__global__ void alpha_synapse(
    int num,
    %(type)s dt,
    int *spike,
    int *Pre,
    %(type)s *Ar,
    %(type)s *Ad,
    %(type)s *Gmax,
    %(type)s *a0,
    %(type)s *a1,
    %(type)s *a2,
    %(type)s *cond )
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    int pre;
    %(type)s ar,ad,gmax;
    %(type)s old_a[3];
    %(type)s new_a[3];

    for( int i=tid; i<num; i+=tot_threads ){
        // copy data from global memory to register
        ar = Ar[i];
        ad = Ad[i];
        pre = Pre[i];
        gmax = Gmax[i];
        old_a[0] = a0[i];
        old_a[1] = a1[i];
        old_a[2] = a2[i];

        // update the alpha function
        new_a[0] = fmax( 0., old_a[0] + dt*old_a[1] );
        new_a[1] = old_a[1] + dt*old_a[2];
        if( spike[pre] )
            new_a[1] += ar*ad;
        new_a[2] = -( ar+ad )*old_a[1] - ar*ad*old_a[0];

        // copy data from register to the global memory
        a0[i] = new_a[0];
        a1[i] = new_a[1];
        a2[i] = new_a[2];
        cond[i] = new_a[0]*gmax;
    }
    return;
}
"""
class AlphaSynapse(BaseSynapse):

    def __init__( self, s_dict, synapse_state, dt, debug=False):
        self.debug = debug
        self.dt = dt
        self.num = len( s_dict['id'] )

        self.pre  = garray.to_gpu( np.asarray( s_dict['pre'], dtype=np.int32 ))
        self.ar   = garray.to_gpu( np.asarray( s_dict['ar'], dtype=np.float64 ))
        self.ad   = garray.to_gpu( np.asarray( s_dict['ad'], dtype=np.float64 ))
        self.gmax = garray.to_gpu( np.asarray( s_dict['gmax'], dtype=np.float64 ))
        self.a0   = garray.zeros( (self.num,), dtype=np.float64 )
        self.a1   = garray.zeros( (self.num,), dtype=np.float64 )
        self.a2   = garray.zeros( (self.num,), dtype=np.float64 )
        self.cond = synapse_state

        self.update = self.get_gpu_kernel()

    @property
    def synapse_class(self): return int(0)

    def update_state(self, buffer, st = None):
        self.update.prepared_async_call(
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num,\
            self.dt,\
            buffer.spike_buffer.gpudata,\
            self.pre.gpudata,\
            self.ar.gpudata,\
            self.ad.gpudata,\
            self.gmax.gpudata,\
            self.a0.gpudata,\
            self.a1.gpudata,\
            self.a2.gpudata,\
            self.cond)

    def get_gpu_kernel(self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        # cuda_src = open('./alpha_synapse.cu','r')
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64)},\
                options=["--ptxas-options=-v"])
        func = mod.get_function("alpha_synapse")
        func.prepare( [ np.int32,   # syn_num
                        np.float64, # dt
                        np.intp,    # spike list
                        np.intp,    # pre-synaptic neuron list
                        np.intp,    # ar array
                        np.intp,    # ad array
                        np.intp,    # gmax array
                        np.intp,    # a0 array
                        np.intp,    # a1 array
                        np.intp,    # a2 array
                        np.intp ] ) # cond array
        return func
