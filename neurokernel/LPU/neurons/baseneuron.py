#!/usr/bin/env python

"""
Base neuron class used by LPU.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
import os.path
import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

class BaseNeuron(object):
    __metaclass__ = ABCMeta

    def __init__(self, n_dict, neuron_state_pointer, dt, debug, LPU_id=None):
        '''
        Every neuron class should setup GPU data structure needed
        by it during initialization. In addition, graded potential neurons
        should also update the neuron_state structures with their
        initial state during  initialization.

        n_dict is a dictionary representing the parameters needed
        by the neuron class.
        For example, if a derived neuron class called IAF needs a
        parameter called bias, n_dict['bias'] will be vector containing
        the bias values fol all the IAF neurons in a particular LPU.

        In addition to the neuron parameters, n_dict will also contain:-
        1. n_dict['cond_pre'] representing the conductance based synapses
           with connection to neurons represented by this class.
        2. n_dict['cond_post'] denoting the neuron indices the synapses mentioned
           above enervate to.
        3. n_dict['reverse'] containing the reverse potentials for the
           conductance based synapses.
        4. n_dict['num_dendrites_cond'] representing the number of dendrites for
           neuron in this class of the conductance type. For eg,
           n_dict['num_dendrites_cond'][0] will represent the number of
           conductance based synapses connecting to the neuron having index 0
           in this object
        5. n_dict['I_pre'] representing the indices of the non-conductance
           based synapses with connections to neurons represented by this
           object , eg:- synapses modelled by filters. This is also includes
           any external input to neurons of this class.
        6. n_dict['I_post'] representing the indices of the neurons the above
           mentioned synapses enervate to.
        7. n_dict['num_dendrites_I'] representing the number of dendrites for
           neuron in this class of the non-conductance type. For eg,
           n_dict['num_dendrites_I'][0] will represent the number of
           non conductance based synapses connecting to the neuron havinf index 0
           in this object.

        Note that you only need the above information if you plan to override the
        default update_I method.

        neuron_state_pointer is an integer representing the inital memory location
        on the GPU for storing the neuron states for this object.
        For graded potential neurons, the data type will be double whereas for
        spiking neurons, it will be int.

        dt represents one time step.

        debug is a boolean and is intended to be used for debugging purposes.

        '''
        self.__neuron_state_pointer = neuron_state_pointer
        self.__num_neurons = len(n_dict['id'])
        _num_dendrite_cond = np.asarray([n_dict['num_dendrites_cond'][i] \
                                    for i in range(self.__num_neurons)], \
                                    dtype=np.int32).flatten()
        _num_dendrite = np.asarray([n_dict['num_dendrites_I'][i] \
                                    for i in range(self.__num_neurons)], \
                                   dtype=np.int32).flatten()

        self.__cum_num_dendrite = garray.to_gpu(np.concatenate(( \
                                    np.asarray([0,], dtype=np.int32), \
                                    np.cumsum(_num_dendrite, dtype=np.int32))))
        self.__cum_num_dendrite_cond = garray.to_gpu(np.concatenate(( \
                                    np.asarray([0,], dtype=np.int32), \
                                    np.cumsum(_num_dendrite_cond, dtype=np.int32))))
        self.__num_dendrite = garray.to_gpu(_num_dendrite)
        self.__num_dendrite_cond = garray.to_gpu(_num_dendrite_cond)
        self.__pre = garray.to_gpu(np.asarray(n_dict['I_pre'], dtype=np.int32))
        self.__cond_pre = garray.to_gpu(np.asarray(n_dict['cond_pre'],
                                                  dtype=np.int32))
        self.__V_rev = garray.to_gpu(np.asarray(n_dict['reverse'],
                                               dtype=np.double))
        self.I = garray.zeros(self.__num_neurons, np.double)
        self.__update_I_cond = self.__get_update_I_cond_func()
        self.__update_I_non_cond = self.__get_update_I_non_cond_func()
        self.__LPU_id = LPU_id
        self.__debug = debug
        if self.__debug:
            if self.__LPU_id is None:
                self.__LPU_id = "default_LPU"
            i = 0
            while os.path.isfile(self.__LPU_id + "_I_" + self.__class__.__name__ + str(i) + ".h5"):
                i+=1
            self.__I_file = tables.openFile(self.__LPU_id + "_I_" +  self.__class__.__name__ +  str(i) + ".h5", mode="w")
            self.__I_file.createEArray("/","array", \
                                     tables.Float64Atom(), (0,self.num_neurons))
            
    @abstractmethod
    def eval(self):
        '''
        This method should update the neuron states. A pointer to
        the start of the memory located will be provided at time of
        initialization.

        self.I.gpudata will be a pointer to the memory location
        where the input current to all the neurons at each step is updated
        if the child class does not override update_I() method
        '''
        pass


    @property
    def neuron_class(self):
        '''
        For future use
        '''
        return 0


    @property
    def update_I_override(self): return False

    def update_I(self, synapse_state, st=None, logger=None):
        '''
        This method should compute the input current to each neuron
        based on the synapse states.
        synapse_state may either contain conductances or currents.
        synapse_state will be an integer representing the inital memory
        location on the GPU reserved for the synapse states. The data
        type for synapse states will be double.
        The information needed to compute the currents is provided in the
        dictionary n_dict at initialization.

        BaseNeuron provides an implementation of this method. To use a
        different implementation, this method should be overrided and
        update_I_override property must be defined to be True in the derived class.

        '''
        self.I.fill(0)
        if self.__pre.size>0:
            self.__update_I_non_cond.prepared_async_call(self.__grid_get_input,\
                self.__block_get_input, st, int(synapse_state), \
                self.__cum_num_dendrite.gpudata, self.__num_dendrite.gpudata, self.__pre.gpudata,
                self.I.gpudata)
        if self.__cond_pre.size>0:
            self.__update_I_cond.prepared_async_call(self.__grid_get_input,\
                self.__block_get_input, st, int(synapse_state), \
                self.__cum_num_dendrite_cond.gpudata, self.__num_dendrite_cond.gpudata,
                self.__cond_pre.gpudata, self.I.gpudata, int(self.__neuron_state_pointer), \
                self.__V_rev.gpudata)
        if self.debug:
            self.__I_file.root.array.append(self.I.get().reshape((1,-1)))
        

    def post_run(self):
        '''
        This method will be called at the end of the simulation.
        '''
        
    def __post_run(self):
        '''
        This private function is used to close the current output file
        when baseneuron is used to compute input current to a neuron and
        the debug flag is set.
        '''
        if self.__debug:
            self.__I_file.close()

    def __get_update_I_cond_func(self):
        template = """
        #define N 32
        #define NUM_NEURONS %(num_neurons)d

        __global__ void get_input(double* synapse, int* cum_num_dendrite, int* num_dendrite, int* pre, double* I_pre, double* V, double* V_rev)
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

            neuron = bid * N + tidy ;
            if(neuron < NUM_NEURONS){
               int n_den = num_den[tidy];
               int start = den_start[tidy];
               double VV = V_in[tidy];


               for(int i = tidx; i < n_den; i += N)
               {
                   input[tidy][tidx] += synapse[pre[start + i]] * (VV - V_rev[start + i]);
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
                       I_pre[neuron] -= input[tidx][0];
                    }
               }

        }
        //can be improved
        """
        mod = SourceModule(template % {"num_neurons": self.__num_neurons}, options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.__block_get_input = (32,32,1)
        self.__grid_get_input = ((self.__num_neurons - 1) / 32 + 1, 1)
        return func

    def __get_update_I_non_cond_func(self):
        template = """
        #define N 32
        #define NUM_NEURONS %(num_neurons)d

        __global__ void get_input(double* synapse, int* cum_num_dendrite, int* num_dendrite, int* pre, double* I_pre)
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

            neuron = bid * N + tidy ;
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
        mod = SourceModule(template % {"num_neurons": self.__num_neurons}, options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp])
        return func
