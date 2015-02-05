from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class HH_PH(BaseNeuron):
    def __init__(self, n_dict, V, dt, debug=False):

        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-4)), 1)
        self.debug = debug

        self.ddt = dt / self.steps

        self.V = V

        self.sa = garray.to_gpu(np.asarray(n_dict['init_sa'], dtype=np.float64))
        self.si = garray.to_gpu(np.asarray(n_dict['init_si'], dtype=np.float64))
        self.dra = garray.to_gpu(np.asarray(n_dict['init_dra'], dtype=np.float64))
        self.dri = garray.to_gpu(np.asarray(n_dict['init_dri'], dtype=np.float64))
        
        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['initV'], 
                         dtype=np.double))
        self.update = self.get_euler_kernel()


    @property
    def neuron_class(self): return True

    def eval(self, st=None):
        self.update.prepared_async_call(
            self.update_grid, self.update_block, st, self.V, self.sa.gpudata,
            self.si.gpudata, self.dra.gpudata, self.dri.gpudata,
            self.I.gpudata, self.num_neurons, self.ddt, self.steps)


    def get_euler_kernel(self):
        template = """
#define E_K (-85)
#define E_Cl (-30)
#define G_s 1.6
#define G_dr 3.5
#define G_Cl 0.056
#define G_K 0.082
#define C 4

__global__ void
hh(%(type)s* d_V, %(type)s* d_sa, %(type)s* d_si,
%(type)s* d_dra, %(type)s* d_dri, %(type)s* I_pre,
int num_neurons, %(type)s dt,
int nsteps)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < num_neurons)
    {
        %(type)s I = I_all[tid];
        %(type)s V = 1000*d_V[tid];  //[V -> mV]
        %(type)s sa = d_sa[tid];
        %(type)s si = d_si[tid];
        %(type)s dra = d_dra[tid];
        %(type)s dri = d_dri[tid];

        %(type)s x_inf, tau_x, dx;

        for(int i = 0; i < nsteps; ++i)
        {
            /* The precision of power constant affects the result */
            x_inf = pow%(fletter)s(1/(1+exp%(fletter)s((-30-V)/13.5)), 1.0/3);
            tau_x = 0.13+3.39*exp%(fletter)s(-(-73-V)*(-73-V)/400);
            dx = (x_inf - sa)/tau_x;
            sa += ddt * dx;

            x_inf = 1/(1+exp%(fletter)s((-55-V)/-5.5));
            tau_x = 113*exp(-(-71-V)*(-71-V)/841);
            dx = (x_inf - si)/tau_x;
            si += ddt * dx;

            x_inf = sqrt%(fletter)s(1/(1+exp%(fletter)s((-5-V)/9)));
            tau_x = 0.5+5.75*exp%(fletter)s(-(-25-V)*(-25-V)/1024);
            dx = (x_inf - dra)/tau_x;
            dra += ddt * dx;

            x_inf = 1/(1+exp%(fletter)s((-25-V)/-10.5));
            tau_x = 890;
            dx = (x_inf - dri)/tau_x;
            dri += ddt * dx;

            dx = (I - G_K*(V-E_K) - G_Cl * (V-E_Cl) - G_s * sa * si * (V-E_K) -
                  G_dr * dra * dri * (V-E_K) - 0.093*(V-10) ) /C;
            V += ddt * dx;
        }
        d_V[tid] = 0.001*V;
        d_sa[tid] = sa;
        d_si[tid] = si;
        d_dra[tid] = dra;
        d_dri[tid] = dri;
    }
}
"""#Used 53 registers, 388 bytes cmem[0], 304 bytes cmem[2]
    #float: Used 35 registers, 380 bytes cmem[0], 96 bytes cmem[2]
        dtype = np.double
        scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
        self.update_block = (128, 1, 1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                                   "fletter": 'f' if scalartype == np.float32 else ''},
                       options = ["--ptxas-options=-v"])
        func = mod.get_function('hh')
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp,
                      np.int32, scalartype, np.int32])
        return func
