from basesynapse import BaseSynapse

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class power_gpot_gpot(BaseSynapse):

    def __init__(self, s_dict,synapse_state_pointer, dt, debug=False):
        self.debug = debug
        self.synapse_state_pointer = synapse_state_pointer
        self.pre = garray.to_gpu(np.asarray(s_dict['pre'], dtype = np.int32))
        self.threshold = garray.to_gpu(np.asarray(s_dict['threshold'],
                                                  dtype = np.double))
        self.slope = garray.to_gpu(np.asarray(s_dict['slope'],
                                                  dtype = np.double))
        self.power = garray.to_gpu(np.asarray(s_dict['power'],
                                                  dtype = np.double))
        self.saturation = garray.to_gpu(np.asarray(s_dict['saturation'],
                                                  dtype = np.double))
        self.delay = garray.to_gpu(np.round(np.asarray(s_dict['delay']) \
                                            * 1e-3 / dt).astype(np.int32))
        self.num_synapse = len(s_dict['id'])

        self.update_func = self.get_update_func()

    @property
    def synapse_class(self): return int(3)


    def update_state(self, buffer, st = None):
        self.update_func.prepared_async_call(self.grid, self.block, st, buffer.gpot_buffer.gpudata, buffer.gpot_buffer.ld, buffer.gpot_current, buffer.gpot_delay_steps, self.pre.gpudata, self.synapse_state_pointer, self.threshold.gpudata, self.slope.gpudata, self.power.gpudata, self.saturation.gpudata, self.delay.gpudata)


    def get_update_func(self):
        template = """
        #define N_synapse %(n_synapse)d

        __global__ void update_gpot_terminal_synapse(double* buffer, int buffer_ld, int current, int delay_steps, int* pre_neuron, double* conductance, double* thres, double* slope, double* power, double* saturation, int* delay)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int total_threads = gridDim.x * blockDim.x;

            int pre;
            double mem;
            int dl;
            int col;

            for(int i = tid; i < N_synapse; i += total_threads)
            {
                pre = pre_neuron[i];
                dl = delay[i];
                col = current - dl;
                if(col < 0)
                {
                    col = delay_steps + col;
                }

                mem = buffer[col * buffer_ld + pre];

                conductance[i] = fmin(saturation[i], slope[i] * pow(fmax(0.0, mem - thres[i]), power[i]));
            }

        }
        """
        #Used 14 registers, 64 bytes cmem[0], 4 bytes cmem[16]
        mod = SourceModule(template % {"n_synapse": self.num_synapse}, options = ["--ptxas-options=-v"])
        func = mod.get_function("update_gpot_terminal_synapse")
        func.prepare([np.intp, np.int32, np.int32, np.int32, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.block = (256,1,1)
        self.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, (self.num_synapse-1) / 256 + 1), 1)
        return func
