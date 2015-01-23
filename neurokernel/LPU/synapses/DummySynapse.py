from basesynapse import BaseSynapse

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_src = """
__global__ void dummy_synapse(
    %(type)s *buffer,
    int buffer_ld,
    int buffer_curr,
    int buffer_delay_steps,
    int syn_num,
    int *pre_neu_idx,
    int *delay,
    %(type)s *syn_state )
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    int pre;
    int dl;
    int col;

    for( int i=tid; i<syn_num; i+=tot_threads ){
        dl = delay[i];
        col = buffer_curr - dl;
        if( col < 0 )
            col += buffer_delay_steps;
        pre = pre_neu_idx[i];
        syn_state[i] = buffer[ col*buffer_ld+pre ];
    }
    return;
}
"""
class DummySynapse(BaseSynapse):

    def __init__( self, s_dict, synapse_state, dt, debug=False):
        self.debug = debug
        #self.dt = dt
        self.num = len( s_dict['id'] )

        if s_dict.has_key( 'delay' ):
            self.delay = garray.to_gpu(np.round(np.asarray( s_dict['delay'])*1e-3/dt ).astype(np.int32) )
        else:
            self.delay = garray.zeros( self.num, dtype=np.int32 )

        self.pre   = garray.to_gpu( np.asarray( s_dict['pre'], dtype=np.int32 ))
        self.state = synapse_state

        self.update = self.get_gpu_kernel()

    @property
    def synapse_class(self): return int(0)

    def update_state(self, buffer, st = None):
        self.update.prepared_async_call(
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            buffer.gpot_buffer.gpudata,\
            buffer.gpot_buffer.ld,
            buffer.gpot_current,
            buffer.gpot_delay_steps,
            self.num,\
            self.pre.gpudata,\
            self.delay.gpudata,\
            self.state)

    def get_gpu_kernel(self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64)},\
                options=["--ptxas-options=-v"])
        func = mod.get_function("dummy_synapse")
        func.prepare( [ np.intp,    # neuron state buffer
                        np.int32,   # buffer width
                        np.int32,   # buffer position
                        np.int32,   # buffer delay steps
                        np.int32,   # syn_num
                        np.intp,    # pre-synaptic neuron list
                        np.intp,    # delay step
                        np.intp ] ) # cond array
        return func
