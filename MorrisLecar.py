import pycuda.driver as cuda
import numpy as np
import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
import time
import parray
from pycuda.tools import dtype_to_ctype

# In Yiyin, this class is called 'vector_neurons'
class MorrisLecar:
    def __init__(self, num_neurons, num_types, num_cart, neuron_start, dt, num_dendrite,
                 V, n, V1, V2, V3, V4, Tphi, offset, non_input_start):
        """
        Set Morris Lecar neurons in the network.

        Parameters
        ----------
        N : int
            Number of neurons to be added.
        """
        self.dtype = np.double
        self.num_cart = num_cart
        self.num_types = num_types
        self.num_neurons = num_neurons
        self.dt = dt
        self.steps = max(int(round(dt / 1e-5)), 1)

        self.ddt = dt / self.steps

        self.V = garray.to_gpu(V)
        self.n = garray.to_gpu(n)

        self.I_pre = garray.zeros(self.num_neurons, np.double)

        self.h_V = cuda.pagelocked_empty((self.num_types, self.num_cart),
                                         np.double)

        self.cum_num_dendrite = garray.to_gpu(np.concatenate((np.asarray([0, ], dtype = np.int32), np.cumsum(num_dendrite, dtype = np.int32))))
        self.num_dendrite = garray.to_gpu(num_dendrite)
        self.num_input = int(neuron_start[non_input_start])

        self.update = self.get_euler_kernel(neuron_start, V1, V2, V3, V4,
                                            Tphi, offset)
        self.get_input = self.get_input_func()

    def update_I_pre_input(self, I_ext):
        cuda.memcpy_dtod(int(self.I_pre.gpudata), I_ext,
                self.num_input * self.I_pre.dtype.itemsize)

    def read_synapse(self, conductance, V_rev, st = None):
        self.get_input.prepared_async_call(self.grid_get_input,
                                           self.block_get_input, st,
                                           conductance.gpudata,
                                           self.cum_num_dendrite.gpudata,
                                           self.num_dendrite.gpudata,
                                           self.I_pre.gpudata, self.V.gpudata,
                                           V_rev.gpudata)

    def eval(self, buffer, st = None):
        self.update.prepared_async_call(self.update_grid, self.update_block,
                                        st, self.V.gpudata, self.n.gpudata,
                                        int(buffer.buffer.gpudata) + \
                                        buffer.current * buffer.buffer.ld * \
                                        buffer.buffer.dtype.itemsize,
                                        self.num_neurons, self.I_pre.gpudata,
                                        self.ddt * 1000, self.steps)

    def get_euler_kernel(self, neuron_start, V1, V2, V3, V4, Tphi, offset):
        template = open('euler_kernel.cu', 'r')

        dtype = self.dtype
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128, 1, 1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template.read() % {"type": dtype_to_ctype(dtype),
                                              "ntype": self.num_types,
                                              "nneu": self.update_block[0]},
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("hhn_euler_multiple")

        V1_addr, V1_nbytes = mod.get_global("V_1")
        V2_addr, V2_nbytes = mod.get_global("V_2")
        V3_addr, V3_nbytes = mod.get_global("V_3")
        V4_addr, V4_nbytes = mod.get_global("V_4")
        Tphi_addr, Tphi_nbytes = mod.get_global("Tphi")
        neuron_start_addr, neuron_start_nbytes = mod.get_global("neuron_start")
        offset_addr, offset_nbytes = mod.get_global("offset")

        cuda.memcpy_htod(V1_addr, V1)
        cuda.memcpy_htod(V2_addr, V2)
        cuda.memcpy_htod(V3_addr, V3)
        cuda.memcpy_htod(V4_addr, V4)
        cuda.memcpy_htod(Tphi_addr, Tphi)
        cuda.memcpy_htod(neuron_start_addr, neuron_start)
        cuda.memcpy_htod(offset_addr, offset)

        func.prepare([np.intp, np.intp, np.intp, np.int32, np.intp, scalartype,
                      np.int32])

        return func

    def get_euler_kernel1(self, neuron_start, V1, V2, V3, V4, Tphi, offset):
        template = open('euler_kernel1.cu', 'r')

        dtype = self.dtype
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        mod = SourceModule(template.read() % {"type": dtype_to_ctype(dtype),
                                              "ntype": self.num_types},
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("hhn_euler_multiple")

        V1_addr, V1_nbytes = mod.get_global("V_1")
        V2_addr, V2_nbytes = mod.get_global("V_2")
        V3_addr, V3_nbytes = mod.get_global("V_3")
        V4_addr, V4_nbytes = mod.get_global("V_4")
        Tphi_addr, Tphi_nbytes = mod.get_global("Tphi")
        neuron_start_addr, neuron_start_nbytes = mod.get_global("neuron_start")
        offset_addr, offset_nbytes = mod.get_global("offset")


        cuda.memcpy_htod(V1_addr, V1)
        cuda.memcpy_htod(V2_addr, V2)
        cuda.memcpy_htod(V3_addr, V3)
        cuda.memcpy_htod(V4_addr, V4)
        cuda.memcpy_htod(Tphi_addr, Tphi)
        cuda.memcpy_htod(neuron_start_addr, neuron_start)
        cuda.memcpy_htod(offset_addr, offset)

        func.prepare([np.intp, np.intp, np.intp, np.int32, np.intp, scalartype,
                      np.int32])

        self.update_block = (64, 2, 1)
        self.update_grid = ((self.num_neurons - 1) / 64 + 1, 1)

        return func

    def get_input_func(self):
        template = open('input_func.cu', 'r')

        mod = SourceModule(template.read() % {"num_neurons": self.num_neurons},
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.block_get_input = (32, 32, 1)
        self.grid_get_input = ((self.num_neurons - 1) / 32 + 1, 1)

        return func
