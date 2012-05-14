#!/usr/bin/env python

import pycuda.driver as cuda
import numpy as np
import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
import time
import parray
from pycuda.tools import dtype_to_ctype


class network:
    
    def __init__(self, num_types, num_neurons, num_cart, neuron_start, num_dendrite, num_synapse, pre_neuron, post_neuron, syn_thres, syn_slope, syn_power, syn_saturation, syn_delay, V_rev, dt, V, n, V_1, V_2, V_3, V_4, Tphi, offset, non_input_start):
        
        
        self.num_neurons = num_neurons
        self.dt = dt
        
        delay_steps = int(round(max(syn_delay)* 1e-3 / dt ))
        
        self.buffer = circular_array(num_neurons, delay_steps, V)
        
        self.neurons = vector_neurons(num_neurons, num_types, num_cart, neuron_start, dt, num_dendrite, V, n, V_1, V_2, V_3, V_4, Tphi, offset, non_input_start)
        self.synapses = vector_synapse(num_synapse, pre_neuron, post_neuron, syn_thres, syn_slope, syn_power, syn_saturation, syn_delay, V_rev, dt)
        
        #if num_neurons > 32:
        #self.st1 = [cuda.Stream(),cuda.Stream()]
        #self.st2 = cuda.Stream()
        #else:
        self.st1 = None
        self.st2 = None

    def run_step(self, I_ext, out, put = False):
        
        
        #V_hist = garray.empty((N_steps, self.num_neurons), np.double)
        
        self.neurons.I_pre.fill(0)
        self.neurons.update_I_pre_input(I_ext)
        
        self.neurons.read_synapse(self.synapses.conductance, self.synapses.V_rev)
        
        self.neurons.eval(self.buffer)
        
        
        self.synapses.compute_synapse(self.buffer)
        
        cuda.memcpy_dtoh(out, self.neurons.V.gpudata)
        self.buffer.step()
        
        
        
        


class circular_array:
    def __init__(self, num_neurons, delay_steps, rest):
        self.dtype = np.double
        self.num_neurons = num_neurons
        self.delay_steps = delay_steps
        
        self.buffer = parray.empty((delay_steps, num_neurons),np.double)
        
        d_rest = garray.to_gpu(rest)
        self.current = 0
        
        #initializing V buffer
        for i in range(delay_steps):
            cuda.memcpy_dtod(int(self.buffer.gpudata) + self.buffer.ld * i, d_rest.gpudata, d_rest.nbytes)
        
    def step(self):
        self.current += 1
        if self.current >= self.delay_steps:
            self.current = 0
        
        
        

class vector_neurons:
    def __init__(self, num_neurons, num_types, num_cart, neuron_start, dt, num_dendrite, V, n, V_1, V_2, V_3, V_4, Tphi, offset, non_input_start):
        
        self.dtype = np.double
        self.num_cart = num_cart
        self.num_types = num_types
        self.num_neurons = num_neurons
        self.dt = dt
        self.steps = max(int(round(dt / 1e-5)),1)
        
        self.ddt = dt / self.steps
        
        
        self.V = garray.to_gpu(V)
        self.n = garray.to_gpu(n)
        
        
        
        
        
        
        self.I_pre = garray.zeros(self.num_neurons, np.double)
        
        self.h_V = cuda.pagelocked_empty((self.num_types, self.num_cart), np.double)
        
        
        self.cum_num_dendrite = garray.to_gpu(np.concatenate((np.asarray([0,],dtype=np.int32),np.cumsum(num_dendrite,dtype=np.int32))))
        self.num_dendrite = garray.to_gpu(num_dendrite)
        
        
        self.num_input = int(neuron_start[non_input_start])
        
        self.update = self.get_euler_kernel(neuron_start, V_1, V_2, V_3, V_4, Tphi, offset)
        self.get_input = self.get_input_func()
        

    
    def update_I_pre_input(self, I_ext):
        cuda.memcpy_dtod(int(self.I_pre.gpudata), I_ext, self.num_input * self.I_pre.dtype.itemsize)
    
    def read_synapse(self, conductance, V_rev, st = None):
        self.get_input.prepared_async_call(self.grid_get_input, self.block_get_input, st, conductance.gpudata, self.cum_num_dendrite.gpudata, self.num_dendrite.gpudata, self.I_pre.gpudata, self.V.gpudata, V_rev.gpudata)
        
    def eval(self, buffer, st = None):
        self.update.prepared_async_call(self.update_grid, self.update_block, st, self.V.gpudata, self.n.gpudata, int(buffer.buffer.gpudata) + buffer.current * buffer.buffer.ld * buffer.buffer.dtype.itemsize, self.num_neurons, self.I_pre.gpudata, self.ddt*1000, self.steps)
        
        
    
    def get_euler_kernel(self, neuron_start, V_1, V_2, V_3, V_4, Tphi, offset):
        template = """

    #define NVAR 2
    #define NNEU %(nneu)d //NROW * NCOL
    #define NTYPE %(ntype)d

    #define V_L (-0.5)
    #define V_Ca 1.0
    #define V_K (-0.7)
    #define g_Ca 1.1
    #define g_K 2.0
    #define g_L 0.5
    
    
    __device__ __constant__ %(type)s V_1[NTYPE];
    __device__ __constant__ %(type)s V_2[NTYPE];
    __device__ __constant__ %(type)s V_3[NTYPE];
    __device__ __constant__ %(type)s V_4[NTYPE];
    __device__ __constant__ %(type)s Tphi[NTYPE];
    __device__ __constant__ int neuron_start[NTYPE];
    __device__ __constant__ %(type)s offset[NTYPE];
    
    __device__ int search_for_type(int cart_id)
    {
        int low = 0, high = NTYPE, mid;
        while(low < high)
        {
            mid = ((low+high)>>1);
            if(cart_id >= neuron_start[mid])
            {
                low = mid + 1;
            }else
            {
                high = mid;
            }
        }
        return low-1;
    }
    
    __device__ %(type)s compute_n(%(type)s V, %(type)s n, int type)
    {
        %(type)s n_inf = 0.5 * (1 + tanh((V - V_3[type]) / V_4[type]));
        %(type)s dn = Tphi[type] * cosh(( V - V_3[type]) / (V_4[type]*2)) * (n_inf - n);
        return dn;
    }
    
    __device__ %(type)s compute_V(%(type)s V, %(type)s n, %(type)s I, int type)
    {
        %(type)s m_inf = 0.5 * (1+tanh((V - V_1[type])/V_2[type]));
        %(type)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca * m_inf * (V - V_Ca) + offset[type]);
        return dV;
    }


    __global__ void
    hhn_euler_multiple(%(type)s* g_V, %(type)s* g_n, %(type)s* V_buffer, int num_neurons, %(type)s* I_pre, %(type)s dt, int nsteps)
    {
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + threadIdx.x;
        
        int type = search_for_type(cart_id);
    
        %(type)s I, V, n;

        if(cart_id < num_neurons)
        {
            V = g_V[cart_id];
            I = I_pre[cart_id];
            n = g_n[cart_id];
        }
        
        
        
        %(type)s dV, dn;
        
        for(int i = 0; i < nsteps; ++i)
        {
            dn = compute_n(V, n, type);
            
            dV = compute_V(V, n, I, type);
            
            V += dV * dt;
            n += dn * dt;
        }
        
        
        if(cart_id < num_neurons)
        {
            g_V[cart_id] = V;
            V_buffer[cart_id] = V;
            g_n[cart_id] = n;
        }
        
    }
    """#Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0], 308 bytes cmem[2], 28 bytes cmem[16]
        dtype = self.dtype
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128,1,1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype), "ntype": self.num_types, "nneu": self.update_block[0]}, options=["--ptxas-options=-v"])
        func = mod.get_function("hhn_euler_multiple")
        
        V_1_addr, V_1_nbytes = mod.get_global("V_1")
        V_2_addr, V_2_nbytes = mod.get_global("V_2")
        V_3_addr, V_3_nbytes = mod.get_global("V_3")
        V_4_addr, V_4_nbytes = mod.get_global("V_4")
        Tphi_addr, Tphi_nbytes = mod.get_global("Tphi")
        neuron_start_addr, neuron_start_nbytes = mod.get_global("neuron_start")
        offset_addr, offset_nbytes = mod.get_global("offset")
        
        cuda.memcpy_htod(V_1_addr, V_1)
        cuda.memcpy_htod(V_2_addr, V_2)
        cuda.memcpy_htod(V_3_addr, V_3)
        cuda.memcpy_htod(V_4_addr, V_4)
        cuda.memcpy_htod(Tphi_addr, Tphi)
        cuda.memcpy_htod(neuron_start_addr, neuron_start)
        cuda.memcpy_htod(offset_addr, offset)
        
        func.prepare([np.intp, np.intp, np.intp, np.int32, np.intp, scalartype, np.int32])
        
        
        return func


    def get_euler_kernel1(self, neuron_start, V_1, V_2, V_3, V_4, Tphi, offset):
        template = """

    #define NVAR 2
    #define NNEU 64 //NROW * NCOL
    #define NTYPE %(ntype)d

    #define V_L (-0.5)
    #define V_Ca 1
    #define V_K (-0.7)
    #define g_Ca 1.1
    #define g_K 2
    #define g_L 0.5
    
    
    __device__ __constant__ %(type)s V_1[NTYPE];
    __device__ __constant__ %(type)s V_2[NTYPE];
    __device__ __constant__ %(type)s V_3[NTYPE];
    __device__ __constant__ %(type)s V_4[NTYPE];
    __device__ __constant__ %(type)s Tphi[NTYPE];
    __device__ __constant__ int neuron_start[NTYPE];
    __device__ __constant__ %(type)s offset[NTYPE];
    
    
    __device__ int search_for_type(int cart_id)
    {
        int low = 0, high = NTYPE, mid;
        while(low < high)
        {
            mid = ((low+high)>>1);
            if(cart_id >= neuron_start[mid])
            {
                low = mid + 1;
            }else
            {
                high = mid;
            }
        }
        return low-1;
    }
    
    __device__ %(type)s compute_n(%(type)s V, %(type)s n, int type)
    {
        %(type)s dn = Tphi[type] * cosh(( V - V_3[type]) / (V_4[type]*2)) * 0.5 * ((1 + tanh((V - V_3[type]) / V_4[type])) - n);
        return dn;
    }
    
    __device__ %(type)s compute_V(%(type)s V, %(type)s n, %(type)s I, int type)
    {
        %(type)s m_inf = 0.5 * (1+tanh((V - V_1[type])/V_2[type]));
        %(type)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca * m_inf * (V - V_Ca) + offset[type]);
        return dV;
    }


    __global__ void
    hhn_euler_multiple(%(type)s* V, %(type)s* n, %(type)s* V_buffer, int num_neurons, %(type)s* I_pre, %(type)s dt, int nsteps)
    {
        int neuron_id = threadIdx.x;
        int vid = threadIdx.y;
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + neuron_id;
        
        int type = search_for_type(cart_id);
        
        __shared__ %(type)s tmpstates[NVAR][NNEU];
    
        double I;

        if(cart_id < num_neurons)
        {
            if(vid == 0)
            {
                tmpstates[0][neuron_id] = V[cart_id];
                I = I_pre[min(cart_id, num_neurons)];
            }else if (vid == 1)
            {
                tmpstates[1][neuron_id] = n[cart_id];
            }
        }
        
        __syncthreads();
        
        %(type)s d, tmp;
        tmp = tmpstates[vid][neuron_id];
        
        for(int i = 0; i < nsteps; ++i)
        {
            if(vid == 1)
            {
                d = compute_n(tmpstates[0][neuron_id], tmp, type);
            }
            
            if(vid == 0)
            {
                d = compute_V(tmpstates[0][neuron_id], tmp, I, type);
            }
            
            tmp += d * dt;
            
            __syncthreads();
            
            tmpstates[vid][neuron_id] = tmp; //would have to change it to 1 neuron per thread model
            
            __syncthreads();
        }
        
        if(vid == 0)
        {
            if(cart_id < num_neurons)
            {
                V[cart_id] = tmpstates[0][neuron_id];
                V_buffer[cart_id] = tmpstates[0][neuron_id];
            }
        }else if(vid == 1)
        {
            if(cart_id < num_neurons)
            {
                n[cart_id] = tmpstates[1][neuron_id];
            }
        }
        
    }
    """#Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0], 308 bytes cmem[2], 28 bytes cmem[16]
        dtype = self.dtype
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype), "ntype": self.num_types}, options=["--ptxas-options=-v"])
        func = mod.get_function("hhn_euler_multiple")
        
        V_1_addr, V_1_nbytes = mod.get_global("V_1")
        V_2_addr, V_2_nbytes = mod.get_global("V_2")
        V_3_addr, V_3_nbytes = mod.get_global("V_3")
        V_4_addr, V_4_nbytes = mod.get_global("V_4")
        Tphi_addr, Tphi_nbytes = mod.get_global("Tphi")
        neuron_start_addr, neuron_start_nbytes = mod.get_global("neuron_start")
        offset_addr, offset_nbytes = mod.get_global("offset")
        
        
        cuda.memcpy_htod(V_1_addr, V_1)
        cuda.memcpy_htod(V_2_addr, V_2)
        cuda.memcpy_htod(V_3_addr, V_3)
        cuda.memcpy_htod(V_4_addr, V_4)
        cuda.memcpy_htod(Tphi_addr, Tphi)
        cuda.memcpy_htod(neuron_start_addr, neuron_start)
        cuda.memcpy_htod(offset_addr, offset)
        
        func.prepare([np.intp, np.intp, np.intp, np.int32, np.intp, scalartype, np.int32])
        
        self.update_block = (64,2,1)
        self.update_grid = ((self.num_neurons - 1) / 64 + 1, 1)
        return func

 
    def get_input_func(self):
        template = """
        #define N 32
        #define NUM_NEURONS %(num_neurons)d
        
        __global__ void get_input(double* synapse, int* cum_num_dendrite, int* num_dendrite, double* I_pre, double* V, double* V_rev)
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
            
            int n_den = num_den[tidy];
            int start = den_start[tidy];
            double VV = V_in[tidy];
            
            for(int i = tidx; i < n_den; i += N)
            {
                input[tidy][tidx] += synapse[start + i] * (VV - V_rev[start + i]);
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
                
                if(neuron < NUM_NEURONS)
                {
                    I_pre[neuron] -= input[tidx][0];
                }
            }
            
            
        }
        //can be improved
        """
        #Used 15 registers, 8704+0 bytes smem, 64 bytes cmem[0]
        mod = SourceModule(template % {"num_neurons": self.num_neurons}, options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.block_get_input = (32,32,1)
        self.grid_get_input = ((self.num_neurons - 1) / 32 + 1, 1)
        return func

class vector_synapse:
    def __init__(self, num_synapse, pre_neuron, post_neuron, syn_thres, syn_slope, syn_power, syn_saturation, syn_delay, V_rev, dt):
        

        self.dt = dt
        self.num_synapse = num_synapse
        self.pre_neuron = garray.to_gpu(pre_neuron)
        #self.post_neuron = garray.to_gpu(post_neuron)
        
        self.threshold = garray.to_gpu(syn_thres)
        self.slope = garray.to_gpu(syn_slope)
        self.power = garray.to_gpu(syn_power)
        self.saturation = garray.to_gpu(syn_saturation)
        self.delay = garray.to_gpu(np.round(syn_delay * 1e-3 / dt).astype(np.int32))
        self.conductance = garray.zeros(self.num_synapse, np.double)
        
        self.V_rev = garray.to_gpu(V_rev)
        
        self.update_terminal_synapse = self.get_update_terminal_synapse_func()
        self.mem_tmp = garray.empty(self.num_synapse, np.double)
    
    
    def compute_synapse(self, buffer, st = None):
        self.update_terminal_synapse.prepared_async_call(self.grid_terminal_synapse, self.block_terminal_synapse, st, buffer.buffer.gpudata, buffer.buffer.ld, buffer.current, buffer.delay_steps, self.pre_neuron.gpudata, self.conductance.gpudata, self.threshold.gpudata, self.slope.gpudata, self.power.gpudata, self.saturation.gpudata, self.delay.gpudata, self.mem_tmp.gpudata)
 
    
    def get_update_terminal_synapse_func(self):
        template = """
        #define N_synapse %(n_synapse)d
        
        __global__ void update_terminal_synapse(double* buffer, int buffer_ld, int current, int delay_steps, int* pre_neuron, double* conductance, double* thres, double* slope, double* power, double* saturation, int* delay, double* mem_tmp)
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
                mem_tmp[i] = mem;
                //conductance[i] = fmax(0.0, fmin(saturation[i], slope[i] * pow(mem - thres[i], power[i])));
                conductance[i] = fmin(saturation[i], slope[i] * pow(fmax(0.0, mem - thres[i]), power[i]));
            }
        
        }
        """
        #Used 14 registers, 64 bytes cmem[0], 4 bytes cmem[16]
        mod = SourceModule(template % {"n_synapse": self.num_synapse}, options = ["--ptxas-options=-v"])
        func = mod.get_function("update_terminal_synapse")
        func.prepare([np.intp, np.int32, np.int32, np.int32, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.block_terminal_synapse = (256,1,1)
        self.grid_terminal_synapse = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, (self.num_synapse-1) / 256 + 1), 1)
        return func


