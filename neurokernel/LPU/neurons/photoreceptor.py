from tables.utils import convertToNPAtom2

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
import tables

import matplotlib.pyplot as plt

import neurokernel.LPU.utils.curand as curand
import neurokernel.LPU.utils.simpleio as si

from baseneuron import BaseNeuron


class Photoreceptor(BaseNeuron):
    def __init__(self, n_dict, V, input_dt,
                 debug=False, LPU_id='anon'):

        self.num_neurons = len(n_dict['id'])  # NOT n_dict['num_neurons']
        # num_microvilli must be the same for every photoreceptor so the first
        # one is taken
        self.num_microvilli = int(n_dict['num_microvilli'][0])
        self.dtype = np.double  # TODO pass it as parameter in n_dict
        self.V = V  # output of hh !pointer don't use .gpudata
        self.input_dt = input_dt
        self.run_dt = 1e-4

        self.multiple = int(self.input_dt/self.run_dt)
        assert(self.multiple * self.run_dt == self.input_dt)

        self.debug = debug
        self.record_neuron = debug
        self.record_microvilli = debug
        self.LPU_id = LPU_id

        self.block_transduction = (128, 1, 1)
        self.grid_transduction = (self.num_neurons, 1)
        self.block_hh = (256, 1, 1)
        self.grid_hh = ( (self.num_neurons-1)/self.block_hh[0] + 1, 1)

        self._initialize()

    def _initialize(self):
        self._setup_output()
        self._setup_poisson()
        self._setup_transduction()
        self._setup_hh()

    def _setup_output(self):
        outputfile = self.LPU_id + '_out'
        if self.record_neuron:
            self.outputfile_I = tables.openFile(outputfile+'I.h5', 'w')
            self.outputfile_I.createEArray(
                "/", "array",
                tables.Float64Atom() if self.dtype == np.double else tables.Float32Atom(),
                (0, self.num_neurons))

            self.outputfile_V = tables.openFile(outputfile+'V.h5', 'w')
            self.outputfile_V.createEArray(
                "/", "array",
                tables.Float64Atom() if self.dtype == np.double else tables.Float32Atom(),
                (0, self.num_neurons))

        if self.record_microvilli:
            self.outputfile_X0 = tables.openFile(outputfile+'X0.h5', 'w')
            self.outputfile_X0.createEArray(
                "/", "array",
                tables.Int16Atom(),
                (0, self.num_neurons))

            self.outputfile_X1 = tables.openFile(outputfile+'X1.h5', 'w')
            self.outputfile_X1.createEArray(
                "/", "array",
                tables.Int16Atom(),
                (0, self.num_neurons))

            self.outputfile_X2 = tables.openFile(outputfile+'X2.h5', 'w')
            self.outputfile_X2.createEArray(
                "/", "array",
                tables.Int16Atom(),
                (0, self.num_neurons))

            self.outputfile_X3 = tables.openFile(outputfile+'X3.h5', 'w')
            self.outputfile_X3.createEArray(
                "/", "array",
                tables.Int16Atom(),
                (0, self.num_neurons))

            self.outputfile_X4 = tables.openFile(outputfile+'X4.h5', 'w')
            self.outputfile_X4.createEArray(
                "/", "array",
                tables.Int16Atom(),
                (0, self.num_neurons))

            self.outputfile_X5 = tables.openFile(outputfile+'X5.h5', 'w')
            self.outputfile_X5.createEArray(
                "/", "array",
                tables.Int16Atom(),
                (0, self.num_neurons))

            self.outputfile_X6 = tables.openFile(outputfile+'X6.h5', 'w')
            self.outputfile_X6.createEArray(
                "/", "array",
                tables.Int16Atom(),
                (0, self.num_neurons))

    def _setup_poisson(self, seed = 3000):
        self.randState = curand.curand_setup(
            self.block_transduction[0]*self.num_neurons, seed)
        self.photon_absorption_func = get_photon_absorption_func(self.dtype)

    def _setup_transduction(self):
        self.X = []
        tmp = np.zeros((self.num_neurons, self.num_microvilli * 2), np.int16)
        tmp[:,::2] = 50
        # variables G, Gstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros((self.num_neurons, self.num_microvilli * 2), np.int16)
        # variables PLCstar, Dstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros((self.num_neurons, self.num_microvilli * 2), np.int16)
        # variables Cstar, Tstar
        self.X.append(garray.to_gpu(tmp.view(np.int32)))
        tmp = np.zeros((self.num_neurons, self.num_microvilli), np.int16)
        # variables Mstar
        self.X.append(garray.to_gpu(tmp))

        Xaddress = np.empty(4, np.int64)
        for i in range(4):
            Xaddress[i] = int(self.X[i].gpudata)

        change_ind1 = np.asarray([1, 1, 2, 3, 3, 2, 5, 4, 5, 5, 7, 6, 6],
                                 np.int32) - 1
        change_ind2 = np.asarray([1, 1, 3, 4, 1, 1, 1, 1, 1, 7, 1, 1, 1],
                                 np.int32) - 1
        change1 = np.asarray([0, -1, -1, -1, -1, 1, 1, -1, -1, -2, -1, 1, -1],
                             np.int32)
        change2 = np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                             np.int32)

        self.n_s0 = 202.
        self.ns = self.n_s0
        self.Ans = 200.
        self.tau_ns = 1.


        self.transduction_func = get_transduction_func(
            self.dtype, self.block_transduction[0],
            self.num_microvilli, Xaddress,
            change_ind1, change_ind2,
            change1, change2)

    def _setup_hh(self):
        self.I_all = garray.empty((1, self.num_neurons), self.dtype)
        # self.V = garray.empty((1, self.num_neurons), self.dtype)
        self.hhx = [garray.empty((1, self.num_neurons), self.dtype)
                    for i in range(4)]
        # self.V.fill(-71.1358)
        V_init = np.empty((1, self.num_neurons), dtype=np.double)
        V_init.fill(-0.0711358)
        cuda.memcpy_htod(int(self.V), V_init)

        self.hhx[0].fill(0.3566)
        self.hhx[1].fill(0.9495)
        self.hhx[2].fill(0.0254)
        self.hhx[3].fill(0.9766)
        self.sum_current_func = get_sum_current_func(self.dtype,
                                                     self.block_transduction[0])
        self.hh_func = get_hh_func(self.dtype)

    def _post_run(self):
        if self.record_neuron:
            self.outputfile_I.close()
            self.outputfile_V.close()
        if self.record_microvilli:
            self.outputfile_X0.close()
            self.outputfile_X1.close()
            self.outputfile_X2.close()
            self.outputfile_X3.close()
            self.outputfile_X4.close()
            self.outputfile_X5.close()
            self.outputfile_X6.close()

    def _write_outputfile(self):
        if self.record_neuron:
            self.outputfile_I.root.array.append(self.I_all.get())
            self.outputfile_I.flush()
            self.outputfile_V.root.array.append(self.V.get())
            self.outputfile_V.flush()

        if self.record_microvilli:
            tmp = self.X[0].get().view(np.int16) # G G*
            self.outputfile_X0.root.array.append([tmp[:, 0]])
            self.outputfile_X0.flush()
            self.outputfile_X1.root.array.append([tmp[:, 1]])
            self.outputfile_X1.flush()

            tmp = self.X[1].get().view(np.int16) # PLC* D*
            self.outputfile_X2.root.array.append([tmp[:,0]])
            self.outputfile_X2.flush()
            self.outputfile_X3.root.array.append([tmp[:,1]])
            self.outputfile_X3.flush()

            tmp = self.X[2].get().view(np.int16) # C* T*
            self.outputfile_X4.root.array.append([tmp[:,0]])
            self.outputfile_X4.flush()
            self.outputfile_X5.root.array.append([tmp[:,1]])
            self.outputfile_X5.flush()

            self.outputfile_X6.root.array.append([self.X[3].get()[:,0]]) # M*
            self.outputfile_X6.flush()

    def eval(self, st=None):
        # self.I is actually pointer to photon input
        self.photon_absorption_func.prepared_call(
            self.grid_transduction, self.block_transduction,
            self.randState.gpudata, self.X[3].gpudata, self.X[3].shape[1],
            self.num_microvilli, self.I.gpudata)


        for _ in range(self.multiple):
            self.transduction_func.prepared_call(
                self.grid_transduction, self.block_transduction,
                self.randState.gpudata, self.X[0].shape[1], self.run_dt,
                self.V, self.ns)
            self.sum_current_func.prepared_call(
                self.grid_transduction, self.block_transduction,
                self.X[2].gpudata, self.X[2].shape[1], self.num_microvilli,
                self.I_all.gpudata, self.V)

            self.hh_func.prepared_call(
                self.grid_hh, self.block_hh,
                self.I_all.gpudata, self.V, self.hhx[0].gpudata,
                self.hhx[1].gpudata, self.hhx[2].gpudata, self.hhx[3].gpudata,
                self.num_neurons, self.run_dt/10, 10)

    #TODO create a function that runs all functions related to simulation and
    # and does not rely on the user to call the functions in the right order
    def generate_graphs(self, dur):
        """ Access output files and
            process them to generate graphs and/or videos
        """
        outputfile = self.outputfile

        dt = self.run_dt # check if other dt is needed
        Nt = int(dur/dt)
        t = np.arange(0, dt*Nt, dt)

        axis = 1  # axis along which the summation is done
        neuron = 0  # neuron id
        microv = 0  # microvillus id

        input = si.read_array(self.inputfile)
        if self.record_neuron:
            outputV = si.read_array(outputfile + 'V.h5')
            outputIall = si.read_array(outputfile + 'I.h5')
        if self.record_microvilli:
            outputX0 = si.read_array(outputfile + 'X0.h5')
            outputX1 = si.read_array(outputfile + 'X1.h5')
            outputX2 = si.read_array(outputfile + 'X2.h5')
            outputX3 = si.read_array(outputfile + 'X3.h5')
            outputX4 = si.read_array(outputfile + 'X4.h5')
            outputX5 = si.read_array(outputfile + 'X5.h5')
            outputX6 = si.read_array(outputfile + 'X6.h5')


        fig, ax = plt.subplots(5,2, sharex=True)

        ax[0,0].plot(t,input[:,neuron])
        ax[0,0].set_title('Input Light Intensity per second', fontsize=12)

        if self.record_neuron:
            ax[0,1].plot(t,outputV[:,neuron])
            ax[0,1].set_title('Output Potential', fontsize=12)
            ax[1,0].plot(t,outputIall[:,neuron])
            ax[1,0].set_title('Output Current', fontsize=12)
        if self.record_microvilli:
            ax[1,1].plot(t,outputX0[:,neuron])
            ax[1,1].set_title('G state of a microvillus', fontsize=12)
            ax[2,0].plot(t,outputX1[:,neuron])
            ax[2,0].set_title('G star state of a microvillus', fontsize=12)
            ax[2,1].plot(t,outputX2[:,neuron])
            ax[2,1].set_title('PLC star state of a microvillus', fontsize=12)
            ax[3,0].plot(t,outputX3[:,neuron])
            ax[3,0].set_title('D star state of a microvillus', fontsize=12)
            ax[3,1].plot(t,outputX4[:,neuron])
            ax[3,1].set_title('C star state of a microvillus', fontsize=12)
            ax[4,0].plot(t,outputX5[:,neuron])
            ax[4,0].set_title('T star state of a microvillus', fontsize=12)
            ax[4,1].plot(t,outputX6[:,neuron])
            ax[4,1].set_title('M star state of a microvillus', fontsize=12)

        fig.canvas.draw()
        fig.savefig(outputfile + '_plot.png', bbox_inches='tight')

        # Visualizer needs gexf file that is which is not present here

# end of photoreceptor
def get_photon_absorption_func(dtype):
    template = """

#include "curand_kernel.h"
extern "C" {
__global__ void
photon_absorption(curandStateXORWOW_t *state, short* M, int ld,
                  int num_microvilli, %(type)s* input)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;

    %(type)s lambda = input[bid] / num_microvilli;
    //int photon = input[bid];

    int n_photon;

    curandStateXORWOW_t localstate = state[bdim*bid + tid];

    for(int i = tid; i < num_microvilli; i += bdim)
    {
        //M[i + bid * ld] += photon;//curand_poisson(state + tid + bdim*bid, lambda);
        n_photon = curand_poisson(&localstate, lambda);
        if(n_photon)
        {
            M[i + bid * ld] += n_photon;
        }
    }
    state[bdim*bid + tid] = localstate;
}

}
"""
# Used 33 registers, 352 bytes cmem[0], 328 bytes cmem[2]
# float: Used 47 registers, 352 bytes cmem[0], 332 bytes cmem[2]
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       options = ["--ptxas-options=-v"],
                       no_extern_c = True)
    func = mod.get_function('photon_absorption')
    func.prepare([np.intp, np.intp, np.int32, np.int32, np.intp])
    return func


def get_transduction_func(dtype, block_size, num_microvilli, Xaddress,
                          change_ind1, change_ind2, change1, change2):
    template = """

#include "curand_kernel.h"

extern "C" {
#include "stdio.h"

#define NUM_MICROVILLI %(num_microvilli)d
#define BLOCK_SIZE %(block_size)d
#define LA 0.5

/* Simulation Constants */
#define C_T     0.5     /* Total concentration of calmodulin */
#define G_T     50      /* Total number of G-protein */
#define PLC_T   100     /* Total number of PLC */
#define T_T     25      /* Total number of TRP/TRPL channels */
#define I_TSTAR 0.68    /* Average current through one opened TRP/TRPL channel (pA)*/

#define GAMMA_DSTAR     4.0 /* s^(-1) rate constant*/
#define GAMMA_GAP       3.0 /* s^(-1) rate constant*/
#define GAMMA_GSTAR     3.5 /* s^(-1) rate constant*/
#define GAMMA_MSTAR     3.7 /* s^(-1) rate constant*/
#define GAMMA_PLCSTAR   144 /* s^(-1) rate constant */
#define GAMMA_TSTAR     25  /* s^(-1) rate constant */

#define H_DSTAR         37.8    /* strength constant */
#define H_MSTAR         40      /* strength constant */
#define H_PLCSTAR       11.1    /* strength constant */
#define H_TSTARP        11.5    /* strength constant */
#define H_TSTARN        10      /* strength constant */

#define K_P     0.3     /* Dissociation coefficient for calcium positive feedback */
#define K_P_INV 3.3333  /* K_P inverse ( too many decimals are not important) */
#define K_N     0.18    /* Dissociation coefficient for calmodulin negative feedback */
#define K_N_INV 5.5555  /* K_N inverse ( too many decimals are not important) */
#define K_U     30      /* (mM^(-1)s^(-1)) Rate of Ca2+ uptake by calmodulin */
#define K_R     5.5     /* (mM^(-1)s^(-1)) Rate of Ca2+ release by calmodulin */
#define K_CA    1000    /* s^(-1) diffusion from microvillus to somata (tuned) */

#define K_NACA  3e-8    /* Scaling factor for Na+/Ca2+ exchanger model */

#define KAPPA_DSTAR         1300.0  /* s^(-1) rate constant - there is also a capital K_DSTAR */
#define KAPPA_GSTAR         7.05    /* s^(-1) rate constant */
#define KAPPA_PLCSTAR       15.6    /* s^(-1) rate constant */
#define KAPPA_TSTAR         150.0   /* s^(-1) rate constant */
#define K_DSTAR             100.0   /* rate constant */

#define F                   96485   /* (mC/mol) Faraday constant (changed from paper)*/
#define N                   4       /* Binding sites for calcium on calmodulin */
#define R                   8.314   /* (J*K^-1*mol^-1)Gas constant */
#define T                   293     /* (K) Absolute temperature */
#define VOL                 3e-9    /* changed from 3e-12microlitres to nlitres
                                     * microvillus volume so that units agree */

#define N_S0_DIM        1   /* initial condition */
#define N_S0_BRIGHT     2

#define A_N_S0_DIM      4   /* upper bound for dynamic increase (of negetive feedback) */
#define A_N_S0_BRIGHT   200

#define TAU_N_S0_DIM    3000    /* time constant for negative feedback */
#define TAU_N_S0_BRIGHT 1000

#define NA_CO           120     /* (mM) Extracellular sodium concentration */
#define NA_CI           8       /* (mM) Intracellular sodium concentration */
#define CA_CO           1.5     /* (mM) Extracellular calcium concentration */

#define G_TRP           8       /* conductance of a TRP channel */
#define TRP_REV         0       /* TRP channel reversal potential */

__device__ __constant__ long long int d_X[4];
__device__ __constant__ int change_ind1[13];
__device__ __constant__ int change1[13];
__device__ __constant__ int change_ind2[13];
__device__ __constant__ int change2[13];

/* cc = n/(NA*VOL) [6.0221413e+23 mol^-1 * 3*10e-21 m^3] */
__device__ float num_to_mM(int n)
{
    return n * 5.5353e-4; // n/1806.6;
}

/* n = cc*VOL*NA [6.0221413e+23 mol^-1 * 3*10e-21 m^3] */
__device__ float mM_to_num(float cc)
{
    return rintf(cc * 1806.6);
}

/* Assumes Hill constant (=2) for positive calcium feedback */
__device__ float compute_fp(float Ca_cc)
{
    float tmp = Ca_cc*K_P_INV;
    tmp *= tmp;
    return tmp/(1 + tmp);
}

/* Assumes Hill constant(=3) for negative calmodulin feedback */
__device__ float compute_fn(float Cstar_cc, float ns)
{
    float tmp = Cstar_cc*K_N_INV;
    tmp *= tmp*tmp;
    return ns*tmp/(1 + tmp);
}

/* Vm [V] */
__device__ float compute_ca(int Tstar, float Cstar_cc, float Vm)
{
    float I_in = Tstar*G_TRP*fmaxf(-Vm, -TRP_REV);
    /* CaM = C_T - Cstar_cc */
    float denom = (K_CA + (N*K_U*C_T) - (N*K_U)*Cstar_cc + 179.0952 * expf(-(F/(R*T))*Vm));  // (K_NACA*NA_CO^3/VOL*F)
    /* I_Ca ~= 0.4*I_in */
    float numer = (0.4*I_in)/(2*VOL*F) +
                  ((K_NACA*CA_CO*NA_CI*NA_CI*NA_CI)/(VOL*F)) +  // in paper it's -K_NACA... due to different conventions
                  N*K_R*Cstar_cc;

    return fmaxf(1.6e-4, numer/denom);
}

__global__ void
transduction(curandStateXORWOW_t *state, int ld1,
             float dt, %(type)s* d_Vm, float ns)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ int X[BLOCK_SIZE][7];  // number of molecules
    __shared__ float Ca[BLOCK_SIZE];
    __shared__ float Vm;  // membrane voltage, shared over all threads
    __shared__ float fn[BLOCK_SIZE];

    if(tid == 0)
    {
        Vm = d_Vm[bid];  // V
    }

    __syncthreads();


    float sumrate;
    float dt_advanced;
    int reaction_ind;
    short2 tmp;

    // copy random generator state locally to avoid accessing global memory
    curandStateXORWOW_t localstate = state[BLOCK_SIZE*bid + tid];

    // iterate over all microvilli in one photoreceptor
    for(int i = tid; i < NUM_MICROVILLI; i += BLOCK_SIZE)
    {
        // load variables that are needed for computing calcium concentration
        //Ca[tid] = ((%(type)s*)d_X[7])[bid*ld2 + i]; // no need to store calcium
        tmp = ((short2*)d_X[2])[bid*ld1 + i];
        X[tid][5] = tmp.x;
        X[tid][6] = tmp.y;

        // update calcium concentration
        Ca[tid] = compute_ca(X[tid][6], num_to_mM(X[tid][5]), Vm);
        fn[tid] = compute_fn(num_to_mM(X[tid][5]), ns);

        // load the rest of variables
        tmp = ((short2*)d_X[1])[bid*ld1 + i];
        X[tid][4] = tmp.y;
        X[tid][3] = tmp.x;
        tmp = ((short2*)d_X[0])[bid*ld1 + i];
        X[tid][2] = tmp.y;
        X[tid][1] = tmp.x;
        X[tid][0] = ((short*)d_X[3])[bid*ld1 + i];

        // compute total rate of reaction
        sumrate = 0;
        sumrate += mM_to_num(K_U) * Ca[tid] * (0.5 - num_to_mM(X[tid][5]) );  //11
        sumrate += mM_to_num(K_R) * num_to_mM(X[tid][5]);  //12
        sumrate += GAMMA_TSTAR * (1 + H_TSTARN*fn[tid]) * X[tid][6];  // 10
        sumrate += GAMMA_DSTAR * (1 + H_DSTAR*fn[tid]) * X[tid][4];  // 8
        sumrate += GAMMA_PLCSTAR * (1 + H_PLCSTAR*fn[tid]) * X[tid][3];  // 7
        sumrate += GAMMA_MSTAR * (1 + H_MSTAR*fn[tid]) * X[tid][0];  // 1
        sumrate += KAPPA_DSTAR * X[tid][3];  // 6
        sumrate += GAMMA_GAP * X[tid][2] * X[tid][3];  // 4
        sumrate += KAPPA_PLCSTAR * X[tid][2] * (PLC_T-X[tid][3]);  // 3
        sumrate += GAMMA_GSTAR * (G_T - X[tid][2] - X[tid][1] - X[tid][3]);  // 5
        sumrate += KAPPA_GSTAR * X[tid][1] * X[tid][0];  // 2
        sumrate += (KAPPA_TSTAR/(K_DSTAR*K_DSTAR)) *
                   (1 + H_TSTARP*compute_fp( Ca[tid] )) *
                   X[tid][4]*(X[tid][4]-1)*(T_T-X[tid][6])*0.5 ;  // 9

        // choose the next reaction time
        dt_advanced = -logf(curand_uniform(&localstate))/(LA + sumrate);

        // If the reaction time is smaller than dt,
        // pick the reaction and update,
        // then compute the total rate and next reaction time again
        // until all dt_advanced is larger than dt.
        // Note that you don't have to compensate for
        // the last reaction time that exceeds dt.
        // The reason is that the exponential distribution is MEMORYLESS.
        while(dt_advanced <= dt)
        {
            reaction_ind = 0;
            sumrate = curand_uniform(&localstate) * sumrate;

            sumrate -= mM_to_num(K_U) * Ca[tid] * (0.5 - num_to_mM(X[tid][5]) );
            reaction_ind = (sumrate<=2e-5) * 11;

            if(!reaction_ind)
            {
                sumrate -= mM_to_num(K_R) * num_to_mM(X[tid][5]);
                reaction_ind = (sumrate<=2e-5) * 12;
                if(!reaction_ind)
                {
                    sumrate -= GAMMA_TSTAR * (1 + H_TSTARN*fn[tid]) * X[tid][6];
                    reaction_ind = (sumrate<=2e-5) * 10;
                    if(!reaction_ind)
                    {
                        sumrate -= GAMMA_DSTAR * (1 + H_DSTAR*fn[tid]) * X[tid][4];
                        reaction_ind = (sumrate<=2e-5) * 8;

                        if(!reaction_ind)
                        {
                            sumrate -= GAMMA_PLCSTAR * (1 + H_PLCSTAR*fn[tid]) * X[tid][3];
                            reaction_ind = (sumrate<=2e-5) * 7;
                            if(!reaction_ind)
                            {
                                sumrate -= GAMMA_MSTAR * (1 + H_MSTAR*fn[tid]) * X[tid][0];
                                reaction_ind = (sumrate<=2e-5) * 1;
                                if(!reaction_ind)
                                {
                                    sumrate -= KAPPA_DSTAR * X[tid][3];
                                    reaction_ind = (sumrate<=2e-5) * 6;
                                    if(!reaction_ind)
                                    {
                                        sumrate -= GAMMA_GAP * X[tid][2] * X[tid][3];
                                        reaction_ind = (sumrate<=2e-5) * 4;

                                        if(!reaction_ind)
                                        {
                                            sumrate -= KAPPA_PLCSTAR * X[tid][2] * (PLC_T-X[tid][3]);
                                            reaction_ind = (sumrate<=2e-5) * 3;
                                            if(!reaction_ind)
                                            {
                                                sumrate -= GAMMA_GSTAR * (G_T - X[tid][2] - X[tid][1] - X[tid][3]);
                                                reaction_ind = (sumrate<=2e-5) * 5;
                                                if(!reaction_ind)
                                                {
                                                    sumrate -= KAPPA_GSTAR * X[tid][1] * X[tid][0];
                                                    reaction_ind = (sumrate<=2e-5) * 2;
                                                    if(!reaction_ind)
                                                    {
                                                        sumrate -= (KAPPA_TSTAR/(K_DSTAR*K_DSTAR)) *
                                                                   (1 + H_TSTARP*compute_fp( Ca[tid] )) *
                                                                   X[tid][4]*(X[tid][4]-1)*(T_T-X[tid][6])*0.5;
                                                        reaction_ind = (sumrate<=2e-5) * 9;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            int ind;

            // only up to two state variables are needed to be updated
            // update the first one.
            ind = change_ind1[reaction_ind];
            X[tid][ind] += change1[reaction_ind];

            //if(reaction_ind == 9)
            //{
            //    X[tid][ind] = max(X[tid][ind], 0);
            //}

            ind = change_ind2[reaction_ind];
            //update the second one
            if(ind != 0)
            {
                X[tid][ind] += change2[reaction_ind];
            }

            // compute the advance time again
            Ca[tid] = compute_ca(X[tid][6], num_to_mM(X[tid][5]), Vm);
            fn[tid] = compute_fn( num_to_mM(X[tid][5]), ns );
            //fp[tid] = compute_fp( Ca[tid] );

            sumrate = 0;
            sumrate += mM_to_num(K_U) * Ca[tid] * (0.5 - num_to_mM(X[tid][5]) ); //11
            sumrate += mM_to_num(K_R) * num_to_mM(X[tid][5]); //12
            sumrate += GAMMA_TSTAR * (1 + H_TSTARN*fn[tid]) * X[tid][6]; // 10
            sumrate += GAMMA_DSTAR * (1 + H_DSTAR*fn[tid]) * X[tid][4]; // 8
            sumrate += GAMMA_PLCSTAR * (1 + H_PLCSTAR*fn[tid]) * X[tid][3]; // 7
            sumrate += GAMMA_MSTAR * (1 + H_MSTAR*fn[tid]) * X[tid][0]; // 1
            sumrate += KAPPA_DSTAR * X[tid][3]; // 6
            sumrate += GAMMA_GAP * X[tid][2] * X[tid][3]; // 4
            sumrate += KAPPA_PLCSTAR * X[tid][2] * (PLC_T-X[tid][3]);  // 3
            sumrate += GAMMA_GSTAR * (G_T - X[tid][2] - X[tid][1] - X[tid][3]); // 5
            sumrate += KAPPA_GSTAR * X[tid][1] * X[tid][0]; // 2
            sumrate += (KAPPA_TSTAR/(K_DSTAR*K_DSTAR)) *
                       (1 + H_TSTARP*compute_fp( Ca[tid] )) *
                       X[tid][4]*(X[tid][4]-1)*(T_T-X[tid][6])*0.5; // 9

            dt_advanced -= logf(curand_uniform(&localstate))/(LA + sumrate);

        } // end while

        ((short*)d_X[3])[bid*ld1 + i] = X[tid][0];
        ((short2*)d_X[0])[bid*ld1 + i] = make_short2(X[tid][1], X[tid][2]);
        ((short2*)d_X[1])[bid*ld1 + i] = make_short2(X[tid][3], X[tid][4]);
        ((short2*)d_X[2])[bid*ld1 + i] = make_short2(X[tid][5], X[tid][6]);
    }
    // copy the updated random generator state back to global memory
    state[BLOCK_SIZE*bid + tid] = localstate;
}

}
"""
    #ptxas info    : 77696 bytes gmem, 336 bytes cmem[3]
    #ptxas info    : Compiling entry function 'transduction' for 'sm_35'
    #ptxas info    : Function properties for transduction
    #    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    #ptxas info    : Used 60 registers, 7176 bytes smem, 352 bytes cmem[0], 324 bytes cmem[2]
    #float : Used 65 registers, 7172 bytes smem, 344 bytes cmem[0], 168 bytes cmem[2]

    scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(
        template % {
            "type": dtype_to_ctype(dtype),
            "block_size": block_size,
            "num_microvilli": num_microvilli,
            "fletter": 'f' if scalartype == np.float32 else ''
        },
        options = ["--ptxas-options=-v --maxrregcount=56"],
        no_extern_c = True)
    func = mod.get_function('transduction')
    d_X_address, d_X_nbytes = mod.get_global("d_X")
    cuda.memcpy_htod(d_X_address, Xaddress)
    d_change_ind1_address, d_change_ind1_nbytes = mod.get_global("change_ind1")
    d_change_ind2_address, d_change_ind2_nbytes = mod.get_global("change_ind2")
    d_change1_address, d_change1_nbytes = mod.get_global("change1")
    d_change2_address, d_change2_nbytes = mod.get_global("change2")
    cuda.memcpy_htod(d_change_ind1_address, change_ind1)
    cuda.memcpy_htod(d_change_ind2_address, change_ind2)
    cuda.memcpy_htod(d_change1_address, change1)
    cuda.memcpy_htod(d_change2_address, change2)

    func.prepare([np.intp, np.int32, np.float32, np.intp, np.float32])
    func.set_cache_config(cuda.func_cache.PREFER_SHARED)
    return func


def get_hh_func(dtype):
    template = """
#define E_K (-85)
#define E_Cl (-30)
#define G_s 1.6
#define G_dr 3.5
#define G_Cl 0.056
#define G_K 0.082
#define C 4


__global__ void
hh(%(type)s* I_all, %(type)s* d_V, %(type)s* d_sa, %(type)s* d_si,
   %(type)s* d_dra, %(type)s* d_dri, int num_neurons, %(type)s ddt, int multiple)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < num_neurons)
    {
        %(type)s I = I_all[tid] * 1.0;  // rescale the input as temporary remedy
        %(type)s V = 1000*d_V[tid];  //[V -> mV]
        %(type)s sa = d_sa[tid];
        %(type)s si = d_si[tid];
        %(type)s dra = d_dra[tid];
        %(type)s dri = d_dri[tid];

        %(type)s x_inf, tau_x, dx;
        %(type)s dt = 1000 * ddt;

        for(int i = 0; i < multiple; ++i)
        {
            /* The precision of power constant affects the result */
            x_inf = pow%(fletter)s(1/(1+exp%(fletter)s((-30-V)/13.5)), 1.0/3);
            tau_x = 0.13+3.39*exp%(fletter)s(-(-73-V)*(-73-V)/400);
            dx = (x_inf - sa)/tau_x;
            sa += dt * dx;

            x_inf = 1/(1+exp%(fletter)s((-55-V)/-5.5));
            tau_x = 113*exp(-(-71-V)*(-71-V)/841);
            dx = (x_inf - si)/tau_x;
            si += dt * dx;

            x_inf = sqrt%(fletter)s(1/(1+exp%(fletter)s((-5-V)/9)));
            tau_x = 0.5+5.75*exp%(fletter)s(-(-25-V)*(-25-V)/1024);
            dx = (x_inf - dra)/tau_x;
            dra += dt * dx;

            x_inf = 1/(1+exp%(fletter)s((-25-V)/-10.5));
            tau_x = 890;
            dx = (x_inf - dri)/tau_x;
            dri += dt * dx;

            dx = (I - G_K*(V-E_K) - G_Cl * (V-E_Cl) - G_s * sa * si * (V-E_K) -
                  G_dr * dra * dri * (V-E_K) - 0.093*(V-10) ) /C;
            V += dt * dx;
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
    scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                                   "fletter": 'f' if scalartype == np.float32 else ''},
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('hh')
    func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp,
                  np.int32, scalartype, np.int32])
    return func


def get_sum_current_func(dtype, block_size):
    template = """
#define BLOCK_SIZE %(block_size)d

__global__ void
sum_current(short2* d_Tstar, int ld, int num_microvilli, %(type)s* I_all,
            %(type)s* d_Vm)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ int sum[BLOCK_SIZE];
    sum[tid] = 0;

    for(int i = tid; i < num_microvilli; i += BLOCK_SIZE)
    {
        sum[tid] += d_Tstar[i + bid*ld].y;
    }
    __syncthreads();

    if(tid < 32)
    {
        #pragma unroll
        for(int i = 0; i < BLOCK_SIZE/32; ++i)
        {
            sum[tid] += sum[tid + 32*i];
        }
    }

    if(tid < 16)
    {
        sum[tid] += sum[tid+16];
    }

    if(tid < 8)
    {
        sum[tid] += sum[tid+8];
    }

    if(tid < 4)
    {
        sum[tid] += sum[tid+4];
    }

    if(tid < 2)
    {
        sum[tid] += sum[tid+2];
    }

    if(tid == 0)
    {
        %(type)s Vm = d_Vm[bid];
        %(type)s I_in;
        if(Vm < 0)
        {
            I_in = (sum[tid]+sum[tid+1]) * 8 * (-Vm);
        }else
        {
            I_in = 0;
        }

        I_all[bid] = I_in / 15.7; // convert pA into \muA/cm^2
    }
}
"""#Used 18 registers, 512 bytes smem, 352 bytes cmem[0], 24 bytes cmem[2]
    # float: Used 20 registers, 1024 bytes smem, 352 bytes cmem[0], 24 bytes cmem[2]
    scalartype = dtype.type if isinstance(dtype, np.dtype) else dtype
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                                   "block_size": block_size},
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('sum_current')
    func.prepare([np.intp, np.int32, np.int32, np.intp, np.intp])
    return func
