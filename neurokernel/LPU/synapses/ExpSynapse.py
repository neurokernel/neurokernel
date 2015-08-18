"""
Exponential Synapse Model
"""
from basesynapse import BaseSynapse

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_src = """
__global__ void exponential_synapse(
    int num,
    %(type)s dt,
    int *spike,
    int *Pre,
    %(type)s *Tau,
    %(type)s *A,
    %(type)s *Gmax,
    %(type)s *Eff,
    %(type)s *cond )
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    int pre;
    %(type)s a,tau,gmax,eff,d_eff;

    for( int i=tid; i<num; i+=tot_threads ){
        // copy data from global memory to register
        pre = Pre[i];
        a = A[i];
        tau = Tau[i];
        eff = Eff[i];
        gmax = Gmax[i];

        // update the exponetial function
        d_eff = -eff/tau;
        if( spike[pre] )
           d_eff += (1-eff)*a;
        eff += dt*d_eff;

        // copy data from register to the global memory
        Eff[i] = eff;
        cond[i] = eff*gmax;
    }
    return;
}
"""
class ExpSynapse(BaseSynapse):
    """
    Exponential Decay Synapse
    """
    def __init__( self, s_dict, synapse_state, dt, debug=False):
        self.debug = debug
        self.dt = dt
        self.num = len( s_dict['id'] )

        self.pre  = garray.to_gpu( np.asarray( s_dict['pre'], dtype=np.int32 ))
        self.a    = garray.to_gpu( np.asarray( s_dict['a'], dtype=np.float64 ))
        self.tau  = garray.to_gpu( np.asarray( s_dict['tau'], dtype=np.float64 ))
        self.gmax = garray.to_gpu( np.asarray( s_dict['gmax'], dtype=np.float64 ))
        self.eff  = garray.zeros( (self.num,), dtype=np.float64 )
        self.cond = synapse_state

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
            self.tau.gpudata,\
            self.a.gpudata,\
            self.gmax.gpudata,\
            self.eff.gpudata,\
            self.cond)

    def _get_gpu_kernel(self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        # cuda_src = open('./alpha_synapse.cu','r')
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64)},\
                options=["--ptxas-options=-v"])
        func = mod.get_function("exponential_synapse")
        func.prepare( [ np.int32,   # syn_num
                        np.float64, # dt
                        np.intp,    # spike list
                        np.intp,    # pre-synaptic neuron list
                        np.intp,    # tau; time constant
                        np.intp,    # a; bump size
                        np.intp,    # gmax array
                        np.intp,    # eff; efficacy
                        np.intp ] ) # cond array
        return func
