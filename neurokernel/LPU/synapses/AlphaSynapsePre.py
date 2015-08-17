"""
Alpha Synapse with Pre-Synaptic Innervation
"""
from basesynapse import BaseSynapse

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_src_synapse_kernel = """
#include <math.h>

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
    %(type)s *I,
    %(type)s *cond )
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    int pre;
    %(type)s ar,ad,gmax,presyn;
    %(type)s old_a[3];
    %(type)s new_a[3];

    for( int i=tid; i<num; i+=tot_threads ){
        // copy data from global memory to register
        ar = Ar[i];
        ad = Ad[i];
        pre = Pre[i];
        gmax = Gmax[i];
        presyn = I[i];
        old_a[0] = a0[i];
        old_a[1] = a1[i];
        old_a[2] = a2[i];

        // update the alpha function
        new_a[0] = fmax( 0., old_a[0] + dt*old_a[1] );
        new_a[1] = old_a[1] + dt*old_a[2];
        if( spike[pre] )
            new_a[1] += ar*ad*exp(-presyn); //NOTE: choose between exp and expf
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
cuda_src_synapse_update_I = """
#define N 32
#define NUM %(num)d

__global__ void get_input(
    double* synapse,
    int* cum_num_dendrite,
    int* num_dendrite,
    int* pre,
    double* I_pre)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bid = blockIdx.x;

    int sid;

    __shared__ int num_den[32];
    __shared__ int den_start[32];
    __shared__ double input[32][33];

    if(tidy == 0)
    {
        sid = bid * N + tidx;
        if(sid < NUM)
        {
            num_den[tidx] = num_dendrite[sid];
        }
    } else if(tidy == 1)
    {
        sid = bid * N + tidx;
        if(sid < NUM)
        {
            den_start[tidx] = cum_num_dendrite[sid];
        }
    }

    input[tidy][tidx] = 0.0;

    __syncthreads();

    sid = bid * N + tidy;
    if(sid < NUM){
       int n_den = num_den[tidy];
       int start = den_start[tidy];

       for(int i = tidx; i < n_den; i += N)
       {
           input[tidy][tidx] += synapse[pre[start + i]];
       }
    }


    __syncthreads();

    if(tidy < 8)
    {
        input[tidx][tidy] += input[tidx][tidy + 8];
        input[tidx][tidy] += input[tidx][tidy + 16];
        input[tidx][tidy] += input[tidx][tidy + 24];
    }

    __syncthreads();

    if(tidy < 4)
    {
        input[tidx][tidy] += input[tidx][tidy + 4];
    }

    __syncthreads();

    if(tidy < 2)
    {
        input[tidx][tidy] += input[tidx][tidy + 2];
    }

    __syncthreads();

    if(tidy == 0)
    {
        input[tidx][0] += input[tidx][1];
        sid = bid*N+tidx;
        if(sid < NUM)
        {
            I_pre[sid] += input[tidx][0];
        }
    }
}
//can be improved
"""

class AlphaSynapsePre(BaseSynapse):

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

        _num_dendrite_cond = np.asarray(
            [s_dict['num_dendrites_cond'][i] for i in s_dict['id']],\
            dtype=np.int32).flatten()
        _num_dendrite = np.asarray(
            [s_dict['num_dendrites_I'][i] for i in s_dict['id']],\
            dtype=np.int32).flatten()

        self._cum_num_dendrite = garray.to_gpu(_0_cumsum(_num_dendrite))
        self._cum_num_dendrite_cond = garray.to_gpu(_0_cumsum(_num_dendrite_cond))
        self._num_dendrite = garray.to_gpu(_num_dendrite)
        self._num_dendrite_cond = garray.to_gpu(_num_dendrite_cond)
        self._pre = garray.to_gpu(np.asarray(s_dict['I_pre'], dtype=np.int32))
        self._cond_pre = garray.to_gpu(np.asarray(s_dict['cond_pre'], dtype=np.int32))
        self._V_rev = garray.to_gpu(np.asarray(s_dict['reverse'],dtype=np.double))
        self.I = garray.zeros(self.num, np.double)
        #self._update_I_cond = self._get_update_I_cond_func()
        self._update_I_non_cond = self._get_update_I_non_cond_func()
        self.update = self._get_gpu_kernel()

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
            self.I.gpudata,\
            self.cond)

    def update_I(self, synapse_state, st=None):
        self.I.fill(0.)
        if self._pre.size > 0:
            self._update_I_non_cond.prepared_async_call(
                self._grid_get_input,
                self._block_get_input,
                st,
                int(synapse_state),
                self._cum_num_dendrite.gpudata,
                self._num_dendrite.gpudata,
                self._pre.gpudata,
                self.I.gpudata)

    def _get_gpu_kernel(self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        # cuda_src = open('./alpha_synapse.cu','r')
        mod = SourceModule( \
                cuda_src_synapse_kernel % {"type": dtype_to_ctype(np.float64)},\
                options=["--ptxas-options=-v"])
        func = mod.get_function("alpha_synapse")
        func.prepare([np.int32,   # syn_num
                      np.float64, # dt
                      np.intp,    # spike list
                      np.intp,    # pre-synaptic neuron list
                      np.intp,    # ar array
                      np.intp,    # ad array
                      np.intp,    # gmax array
                      np.intp,    # a0 array
                      np.intp,    # a1 array
                      np.intp,    # a2 array
                      np.intp,    # pre-synaptic input
                      np.intp])   # cond array
        return func

    def _get_update_I_non_cond_func(self):
        mod = SourceModule(\
                cuda_src_synapse_update_I % {"num": self.num},
                options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp,  # synapse state
                      np.intp,  # cumulative dendrites number
                      np.intp,  # dendrites number
                      np.intp,  # pre-synaptic number ID
                      np.intp]) # output

        self._block_get_input = (32,32,1)
        self._grid_get_input = ((self.num - 1) / 32 + 1, 1)
        return func

def _0_cumsum(it, dtype=np.int32):
    """
    Like numpy.cumsum but with 0 at the head of output, i.e.
    [0, np.cumsum(it)]
    """
    return np.concatenate((np.asarray([0,], dtype=dtype),
                           np.cumsum(it, dtype=dtype)))
