from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

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
    def __init__(self, n_dict, spk, dt, debug=False, LPU_id=None):
        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = 1
        self.debug = debug
        self.LPU_id = LPU_id

        self.Vr  = garray.to_gpu( np.asarray( n_dict['Vr'], dtype=np.float64 ))
        self.Vt  = garray.to_gpu( np.asarray( n_dict['Vt'], dtype=np.float64 ))
        self.C   = garray.to_gpu( np.asarray( n_dict['C'], dtype=np.float64 ))
        self.R   = garray.to_gpu( np.asarray( n_dict['R'], dtype=np.float64 ))
        self.V   = garray.to_gpu( np.asarray( n_dict['V'], dtype=np.float64 ))
        self.spk = spk

        _num_dendrite_cond = np.asarray([n_dict['num_dendrites_cond'][i]
                                         for i in range(self.num_neurons)],
                                        dtype=np.int32).flatten()
        _num_dendrite = np.asarray([n_dict['num_dendrites_I'][i]
                                    for i in range(self.num_neurons)],
                                   dtype=np.int32).flatten()

        self._cum_num_dendrite = garray.to_gpu(np.concatenate((
                                    np.asarray([0,], dtype=np.int32),
                                    np.cumsum(_num_dendrite, dtype=np.int32))))
        self._cum_num_dendrite_cond = garray.to_gpu(np.concatenate((
                                    np.asarray([0,], dtype=np.int32),
                                    np.cumsum(_num_dendrite_cond, 
                                              dtype=np.int32))))
        self._num_dendrite = garray.to_gpu(_num_dendrite)
        self._num_dendrite_cond = garray.to_gpu(_num_dendrite_cond)
        self._pre = garray.to_gpu(np.asarray(n_dict['I_pre'], dtype=np.int32))
        self._cond_pre = garray.to_gpu(np.asarray(n_dict['cond_pre'],
                                                  dtype=np.int32))
        self._V_rev = garray.to_gpu(np.asarray(n_dict['reverse'],
                                               dtype=np.double))
        self.I = garray.zeros(self.num_neurons, np.double)
        self._update_I_cond = self._get_update_I_cond_func()
        self._update_I_non_cond = self._get_update_I_non_cond_func()
        self.update = self.get_gpu_kernel()
        if self.debug:
            if self.LPU_id is None:
                self.LPU_id = "anon"
            self.I_file = tables.openFile(self.LPU_id + "_I.h5", mode="w")
            self.I_file.createEArray("/","array",
                                     tables.Float64Atom(), (0,self.num_neurons))
            self.V_file = tables.openFile(self.LPU_id + "_V.h5", mode="w")
            self.V_file.createEArray("/","array",
                                     tables.Float64Atom(), (0,self.num_neurons))
    @property
    def neuron_class(self): return True

    def eval(self, st=None):
        self.update.prepared_async_call(
            self.gpu_grid,
            self.gpu_block,
            st,
            self.num_neurons,
            self.dt,
            self.spk,
            self.V.gpudata,
            self.I.gpudata,
            self.Vt.gpudata,
            self.Vr.gpudata,
            self.R.gpudata,
            self.C.gpudata)
        if self.debug:
            self.I_file.root.array.append(self.I.get().reshape((1, -1)))
            self.V_file.root.array.append(self.V.get().reshape((1, -1)))
            

    def get_gpu_kernel( self):
        self.gpu_block = (128, 1, 1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
        #cuda_src = open( './leaky_iaf.cu','r')
        mod = SourceModule(
                cuda_src % {"type": dtype_to_ctype(np.float64),
                            "nneu": self.gpu_block[0] },
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
        
    def post_run(self):
        if self.debug:
            self.I_file.close()
            self.V_file.close()

    @property
    def update_I_override(self): return True

    def update_I(self, synapse_state, st=None):
        self.I.fill(0)
        if self._pre.size>0:
            self._update_I_non_cond.prepared_async_call(self._grid_get_input,
                self._block_get_input, st, int(synapse_state),
                self._cum_num_dendrite.gpudata, self._num_dendrite.gpudata,
                self._pre.gpudata, self.I.gpudata)
        if self._cond_pre.size>0:
            self._update_I_cond.prepared_async_call(self._grid_get_input,
                self._block_get_input, st, int(synapse_state),
                self._cum_num_dendrite_cond.gpudata, 
                self._num_dendrite_cond.gpudata,
                self._cond_pre.gpudata, self.I.gpudata, self.V.gpudata,
                self._V_rev.gpudata)
        


    def _get_update_I_cond_func(self):
        template = """
        #define N 32
        #define NUM_NEURONS %(num_neurons)d

        __global__ void get_input(double* synapse, int* cum_num_dendrite, 
                                  int* num_dendrite, int* pre, double* I_pre, 
                                  double* V, double* V_rev)
        {
            int tidx = threadIdx.x;
            int tidy = threadIdx.y;
            int bid = blockIdx.x;

            int neuron;

            __shared__ int num_den[32];
            __shared__ int den_start[32];
            __shared__ double V_in[32];
            __shared__ double input[32][33];

            
            if(tidy == 0)
            {
                neuron = bid * N + tidx;
                if(neuron < NUM_NEURONS)
                {
                    num_den[tidx] = num_dendrite[neuron];
                    V_in[tidx] = V[neuron];
                }
            }else if(tidy == 1)
            {
                neuron = bid * N + tidx;
                if(neuron < NUM_NEURONS)
                {
                    den_start[tidx] = cum_num_dendrite[neuron];
                }
            }

            input[tidy][tidx] = 0.0;

            __syncthreads();
            neuron = bid * N + tidy;
            if(neuron < NUM_NEURONS)
            {
                int n_den = num_den[tidy];
                int start = den_start[tidy];
                double VV = V_in[tidy];


                for(int i = tidx; i < n_den; i += N)
                {
                   input[tidy][tidx] += synapse[pre[start + i]] * 
                                        (VV - V_rev[start + i]);
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
                   neuron = bid*N + tidx;
                   if(neuron < NUM_NEURONS)
                   {
                       I_pre[neuron] -= input[tidx][0];
                    }
               }

        }
        //can be improved
        """
        mod = SourceModule(template % {"num_neurons": self.num_neurons}, 
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self._block_get_input = (32,32,1)
        self._grid_get_input = ((self.num_neurons - 1) / 32 + 1, 1)
        return func

    def _get_update_I_non_cond_func(self):
        template = """
        #define N 32
        #define NUM_NEURONS %(num_neurons)d

        __global__ void get_input(double* synapse, int* cum_num_dendrite, 
                                  int* num_dendrite, int* pre, double* I_pre)
        {
            int tidx = threadIdx.x;
            int tidy = threadIdx.y;
            int bid = blockIdx.x;

            int neuron;

            __shared__ int num_den[32];
            __shared__ int den_start[32];
            __shared__ double input[32][33];

            if(tidy == 0)
            {
                neuron = bid * N + tidx;
                if(neuron < NUM_NEURONS)
                {
                    num_den[tidx] = num_dendrite[neuron];
                }
            }else if(tidy == 1)
            {
                neuron = bid * N + tidx;
                if(neuron < NUM_NEURONS)
                {
                    den_start[tidx] = cum_num_dendrite[neuron];
                }
            }

            input[tidy][tidx] = 0.0;

            __syncthreads();

            neuron = bid * N + tidy;
            if(neuron < NUM_NEURONS){
               int n_den = num_den[tidy];
               int start = den_start[tidy];

               for(int i = tidx; i < n_den; i += N)
               {
                   input[tidy][tidx] += synapse[pre[start] + i];
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
                neuron = bid*N+tidx;
                if(neuron < NUM_NEURONS)
                {
                    I_pre[neuron] += input[tidx][0];
                }
            }
        }
        //can be improved
        """
        mod = SourceModule(template % {"num_neurons": self.num_neurons}, 
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp])
        return func
