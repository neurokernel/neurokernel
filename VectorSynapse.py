import pycuda.driver as cuda
import numpy as np
import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
import time
import parray
from pycuda.tools import dtype_to_ctype

# Old name: vector_synapse
class VectorSynapse:
    def __init__(self, num_synapse, pre_neuron, post_neuron, syn_thres,
                 syn_slope, syn_power, syn_saturation, syn_delay, V_rev, dt):
        self.dt = dt
        self.num_synapse = num_synapse
        self.pre_neuron = garray.to_gpu(pre_neuron)
        #self.post_neuron = garray.to_gpu(post_neuron)

        self.threshold = garray.to_gpu(syn_thres)
        self.slope = garray.to_gpu(syn_slope)
        self.power = garray.to_gpu(syn_power)
        self.saturation = garray.to_gpu(syn_saturation)
        self.delay = garray.to_gpu(
                            np.round(syn_delay * 1e-3 / dt).astype(np.int32))
        self.conductance = garray.zeros(self.num_synapse, np.double)

        self.V_rev = garray.to_gpu(V_rev)

        self.update_terminal_synapse = self.get_update_terminal_synapse_func()
        self.mem_tmp = garray.empty(self.num_synapse, np.double)

    def compute_synapse(self, buffer, st = None):
        self.update_terminal_synapse.prepared_async_call(
                                                    self.grid_terminal_synapse,
                                                    self.block_terminal_synapse,
                                                    st, buffer.buffer.gpudata,
                                                    buffer.buffer.ld,
                                                    buffer.current,
                                                    buffer.delay_steps,
                                                    self.pre_neuron.gpudata,
                                                    self.conductance.gpudata,
                                                    self.threshold.gpudata,
                                                    self.slope.gpudata,
                                                    self.power.gpudata,
                                                    self.saturation.gpudata,
                                                    self.delay.gpudata,
                                                    self.mem_tmp.gpudata)

    def get_update_terminal_synapse_func(self):
        template = open('terminal_synapse.cu', 'r')

        mod = SourceModule(template.read() % {"n_synapse": self.num_synapse},
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("update_terminal_synapse")
        func.prepare([np.intp, np.int32, np.int32, np.int32, np.intp, np.intp,
                      np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.block_terminal_synapse = (256, 1, 1)
        self.grid_terminal_synapse = (min(6 * \
                              cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                              (self.num_synapse - 1) / 256 + 1), 1)

        return func
