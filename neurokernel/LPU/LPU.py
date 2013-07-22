#!/usr/bin/env python

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.core import Module
import neurokernel.base as base

from types import *
from collections import Counter

from utils.simpleio import *
import utils.parray as parray
from neurons import *
from synapses import *

class LPU(Module):
    def __init__(self, dt, n_dict_list, s_dict_list, input_file=None,
                 device=None, output_file=None, port_ctrl=base.PORT_CTRL,
                 port_data=base.PORT_DATA, LPU_id=None, debug=False):

        """
        Initialization of a LPU
        
        Parameters
        ----------
        
        dt : double
            one time step.
        n_dict_list : list of dictionaries
            a list of dictionaries describing the neurons in this LPU - one
        dictionary for each type of neuron.
            s_dict_list : list of dictionaries
        a list of dictionaries describing the synapses in this LPU - one
            for each type of synapse.
        input_file : string
            path and name of the input video file
        output_file : string
            path and name of the output files
        port_data : int
            Port to use when communicating with broker.
        port_ctrl : int
            Port used by broker to control module.
        device : int
            Device no to use
        LPU_id : string
            Name of the LPU
        debug : boolean
            This parameter will be passed to all the neuron and synapse
            objects instantiated by this LPU and is intended to be used
            for debugging purposes only.
            Will be set to False by default

        """

        super(LPU, self).__init__(port_data=port_data, port_ctrl=port_ctrl,
                              device=device, id=LPU_id)

        self.dt = dt
        self.debug = debug
        self.device = device
        self.n_dict_list = n_dict_list
        self.s_dict_list = s_dict_list
        self.output_file = output_file
        self.input_file = input_file
        
        if self.output_file is not None:
            self.output = True
        else:
            self.output = False
        self.input_file = input_file

        if self.input_file is not None:
            self.input_eof = False
        
        self._load_neurons()
        self._load_synapses()
        
    @property
    def N_gpot(self): return self.num_public_gpot

    @property
    def N_spike(self): return self.num_public_spike

    def pre_run(self):
        super(LPU,self).pre_run()
        self._setup_connectivity()
        self._init_objects()
        self.buffer = circular_array(self.total_gpot_neurons, \
                        self.gpot_delay_steps, self.V.get()[:self.my_num_gpot_neurons], \
                        self.total_spike_neurons, self.spike_delay_steps)
        
        if self.input_file is not None:
            self.input_h5file = tables.openFile(self.input_file)

            self.one_time_import = 10
            self.file_pointer = 0
            self.I_ext = parray.to_gpu(self.input_h5file.root.real.read(\
                                self.file_pointer, self.file_pointer + \
                                self.one_time_import))
            self.file_pointer += self.one_time_import
            self.frame_count = 0

        if self.output:
            output_file = self.output_file.rsplit('.',1)
            filename = output_file[0]
            if(len(output_file)>1):
                ext = output_file[1]
            else:
                ext = 'h5'

            if self.my_num_gpot_neurons>0:
                self.output_gpot_file = tables.openFile(filename + \
                                                    '_gpot.' + ext , 'w')
                self.output_gpot_file.createEArray("/","array", \
                            tables.Float64Atom(), (0,self.my_num_gpot_neurons))
            if self.my_num_spike_neurons>0:
                self.output_spike_file = tables.openFile(filename + \
                                                    '_spike.' + ext , 'w')
                self.output_spike_file.createEArray("/","array", \
                            tables.Float64Atom(),(0,self.my_num_spike_neurons))
            
    def post_run(self):
        super(LPU,self).post_run()
        if self.output:
            if self.my_num_gpot_neurons > 0:
                self.output_gpot_file.close()
            if self.my_num_spike_neurons > 0:
                self.output.spike_file.close()

        for neuron in self.neurons:
            neuron.post_run()

        for synapse in self.synapses:
            synapse.post_run()



    def run_step(self, in_gpot_dict, in_spike_dict, out_gpot, out_spike):
        super(LPU, self).run_step(in_gpot_dict, in_spike_dict, \
                                  out_gpot, out_spike)

        update = True
        if not self.run_on_myself:
            if (len(in_gpot_dict) +len(in_spike_dict)) == 0:
                update = False
            else:
                self._read_LPU_input(in_gpot_dict, in_spike_dict)

        if update and self.update_resting_potential_history:
            self.buffer.update_other_rest(in_gpot_dict, \
                np.sum(self.num_gpot_neurons), self.num_virtual_gpot_neurons)
            self.update_resting_potential_history = False
            
        if update:
            if self.input_file is not None:
                self._read_external_input()

            for neuron in self.neurons:
                neuron.update_I(self.synapse_state.gpudata)
                neuron.eval()

            self._update_buffer()                  
            for synapse in self.synapses:
                # Maybe only gpot_buffer or spike_buffer should be passed
                # based on the synapse class.
                synapse.update_state(self.buffer)

        if not self.run_on_myself:        
            self._extract_output(out_gpot, out_spike)

        if self.output:
            self._write_output()
            


    def _init_objects(self):
        self.neurons = []
        self.synapses = []
        for i in range(len(self.n_dict_list)):
            self.neurons.append(self._instantiate_neuron(i))

        for i in range(len(self.s_dict_list)):
            self.synapses.append(self._instantiate_synapse(i))
        
    def _setup_connectivity(self):
        n_dict_list = self.n_dict_list
        s_dict_list = self.s_dict_list
        self.num_gpot_neurons = []
        self.num_spike_neurons = []
        self.gpot_idx = []
        self.spike_idx = []
        self.public_neuron_list = []
        self.input_neuron_list = []
        self.public_gpot_list = []
        self.public_spike_list = []
        gpot_delay_steps = 1
        spike_delay_steps = 1
        
        for n_dict in n_dict_list:
            if n_dict['spiking'][0]:
                self.num_gpot_neurons.append(0)
                self.num_spike_neurons.append(len(n_dict['type']))
                self.spike_idx.extend(n_dict['id'])
            else:
                self.num_gpot_neurons.append(len(n_dict['type']))
                self.num_spike_neurons.append(0)
                self.gpot_idx.extend(n_dict['id'])

        
        
        order = np.argsort(np.concatenate((np.asarray(self.gpot_idx, \
                            dtype=np.int32), np.asarray(self.spike_idx, \
                            dtype=np.int32))))
        spike_shift = np.sum(self.num_gpot_neurons)
        synapse_types = []

        count = 0
        
        for n_dict in n_dict_list:
            idx = order[np.asarray(np.where(np.asarray(\
                    n_dict['public'], dtype=bool) == True))+count]
            self.public_neuron_list.extend(idx.flatten())
            if n_dict['spiking'][0]:
                self.public_spike_list.extend(idx.flatten())
            else:
                self.public_gpot_list.extend(idx.flatten())

            input_idx =order[np.asarray(np.where(np.asarray(\
                            n_dict['input'], dtype=bool) == True))+count]
            self.input_neuron_list.extend(input_idx.flatten())
            count += len(n_dict['type'])
                               
        self.input_neuron_list.sort()
        
        for s_dict in s_dict_list:
            s_dict['pre'] = np.asarray(s_dict['pre'], dtype=np.int32)
            s_dict['post'] = np.asarray(s_dict['post'], dtype=np.int32)
            
            s_dict['pre'] = order[s_dict['pre']]
            s_dict['post'] = order[s_dict['post']]
            if s_dict['class'][0] == 0 or s_dict['class'][0] == 1:
                s_dict['pre'] = s_dict['pre'] - spike_shift
            if s_dict['class'][0] == 0 or s_dict['class'][0] == 2:
                s_dict['post'] = s_dict['post'] - spike_shift

            s_dict['pre'] = s_dict['pre'].tolist()
            s_dict['post'] = s_dict['post'].tolist()

            synapse_types.append(s_dict['type'][0])
        
        connectivity = self._conn_dict
        
        
        if len(connectivity)==0:
            self.update_resting_potential_history = False
            self.run_on_myself = True
            self.num_virtual_gpot_neurons = 0
            self.num_virtual_spike_neurons = 0
            self.num_input_gpot_neurons = 0
            self.num_input_spike_neurons = 0
            '''
            Need to complete this
            '''
        else:
            self.update_resting_potential_history = True
            self.run_on_myself = False
            
            num_input_gpot_neurons = np.empty(self.N_input_LPU, np.int32)
            num_input_spike_neurons = np.empty(self.N_input_LPU,np.int32)
            self.virtual_gpot_idx = []
            self.virtual_spike_idx = []
            
            tmp1 = np.sum(num_gpot_neurons)
            tmp2 = np.sum(num_spike_neurons)
            
        
            
            for i,c in enumerate(connectivity.itervalues()):
                other_lpu = c.B_id if self.id == c.A_id else c.A_id
                pre_gpot = c.src_idx(other_lpu, self.LPU_id, src_type='gpot')
            
                num_input_gpot_neurons[i] = len(pre_gpot)

                self.virtual_gpot_idx.append(np.arange(tmp1,tmp1+\
                                    num_input_gpot_neurons[i]).astype(np.int32))
                tmp1 += num_input_gpot_neurons[i]
                
                for j,pre_id in enumerate(pre_gpot):
                    pre_id = int(pre_id)
                    post_idx = c.dest_idx(other_lpu, self.LPU_id, \
                                          'gpot', 'gpot', src_ports=int(pre_id))
                    for post_id in post_idx:
                        post_id = int(post_id)
                        num_syn = c.multapses(other_lpu, 'gpot', pre_id, self.id,
                                              'gpot', post_id)
                        for conn in range(num_syn):
                            s_type = c.get(other_lpu, pre_id, 'gpot', self.id,
                                           'gpot', post_id, conn=conn,
                                           param='type')
                            try:
                                s_id = synapse_types.index(s_type)
                            except ValueError:
                                '''
                                Need the connectivity class to implement this
                                It should return an dictionary with keys
                                representing the synapse parameters, excluding
                                pre/src and post/dest. It need not contain
                                any items.
                                '''
                                s = dict.fromkeys(c.get_dict_params(s_type))
                                s_id = len(synapse_types)
                                for key in s.iterkeys():
                                    s[key] = []
                                synapse_types.append(s_type)
                                s_dict_list.append(s)
                                
                            s_dict_list[s_id]['pre'].append(\
                                        self.vitual_gpot_idx[i][j])
                            s_dict_list[s_id]['post'].append(\
                                        order[self.public_neuron_list[post_id]])
                            for key in s_dict_list[s_id].iterkeys():
                                s_dict_list[s_id][key].append(\
                                    c.get(other_lpu, pre_id, 'gpot', self.id,
                                          'gpot', post_id, conn=conn,
                                          param=key))
                                
                    post_idx = c.dest_idx(other_lpu, self.LPU_id, \
                                          'gpot', 'spike', src_ports=int(pre_id))
                    for post_id in post_idx:
                        post_id = int(post_id)
                        num_syn = c.multapses(other_lpu, 'gpot', pre_id, self.id,
                                              'spike', post_id)
                        for conn in range(num_syn):
                            s_type = c.get(other_lpu, pre_id, 'gpot', self.id,
                                           'spike', post_id, conn=conn,
                                           param='type')
                            try:
                                s_id = synapse_types.index(s_type)
                            except ValueError:
                                s = dict.fromkeys(c.get_dict_params(s_type))
                                s_id = len(synapse_types)
                                for key in s.iterkeys():
                                    s[key] = []
                                synapse_types.append(s_type)
                                s_dict_list.append(s)

                            s_dict_list[s_id]['pre'].append(\
                                        self.vitual_gpot_idx[i][j])
                            s_dict_list[s_id]['post'].append(\
                                        order[self.public_neuron_list[post_id]])\
                                        - spike_shift
                            for key in s_dict_list[s_id].iterkeys():
                                s_dict_list[s_id][key].append(\
                                    c.get(other_lpu, pre_id, 'gpot', self.id,
                                          'spike', post_id, conn=conn,
                                          param=key))
                
                
                pre_spike = c.src_idx(other_lpu, \
                                      self.id, src_type='spike')
                num_input_spike_neurons[i] = len(pre_spike)
                self.virtual_spike_idx.append(np.arange(tmp2,tmp2+\
                    num_input_spike_neurons[i]).astype(np.int32))
                tmp2 += num_input_spike_neurons[i]
                
                for j,pre_id in enumerate(pre_spike):
                    pre_id = int(pre_id)
                    post_idx = c.dest_idx(other_lpu, self.LPU_id, \
                                          'spike', 'gpot', src_ports=int(pre_id))
                    for post_id in post_idx:
                        post_id = int(post_id)
                        num_syn = c.multapses(other_lpu, 'spike', pre_id, self.id,
                                              'gpot', post_id)
                        for conn in range(num_syn):
                            s_type = c.get(other_lpu, pre_id, 'spike', self.id,
                                           'gpot', post_id, conn=conn,
                                           param='type')
                            try:
                                s_id = synapse_types.index(s_type)
                            except ValueError:
                                s = dict.fromkeys(c.get_dict_params(s_type))
                                s_id = len(synapse_types)
                                for key in s.iterkeys():
                                    s[key] = []
                                synapse_types.append(s_type)
                                s_dict_list.append(s)

                            s_dict_list[s_id]['pre'].append(\
                                        self.vitual_spike_idx[i][j])
                            s_dict_list[s_id]['post'].append(\
                                        order[self.public_neuron_list[post_id]])
                            for key in s_dict_list[s_id].iterkeys():
                                s_dict_list[s_id][key].append(\
                                    c.get(other_lpu, pre_id, 'spike', self.id,
                                          'gpot', post_id, conn=conn,
                                          param=key))
                
                    post_idx = c.dest_idx(other_lpu, self.LPU_id, \
                                          'spike', 'spike', src_ports=int(pre_id))
                    for post_id in post_idx:
                        post_id = int(post_id)
                        num_syn = c.multapses(other_lpu, 'spike', pre_id, self.id,
                                              'spike', post_id)
                        for conn in range(num_syn):
                            s_type = c.get(other_lpu, pre_id, 'spike', self.id,
                                           'spike', post_id, conn=conn,
                                           param='type')
                            try:
                                s_id = synapse_types.index(s_type)
                            except ValueError:
                                s = dict.fromkeys(c.get_dict_params(s_type))
                                s_id = len(synapse_types)
                                for key in s.iterkeys():
                                    s[key] = []
                                synapse_types.append(s_type)
                                s_dict_list.append(s)

                            s_dict_list[s_id]['pre'].append(\
                                        self.vitual_spike_idx[i][j])
                            s_dict_list[s_id]['post'].append(\
                                        order[self.public_neuron_list[post_id]])\
                                        - spike_shift
                            for key in s_dict_list[s_id].iterkeys():
                                s_dict_list[s_id][key].append(\
                                    c.get(other_lpu, pre_id, 'spike', self.id,
                                          'spike', post_id, conn=conn,
                                          param=key))
                
            
            
            self.num_input_gpot_neurons = num_input_gpot_neurons
            self.num_input_spike_neurons = num_input_spike_neurons
            
            # total number of input graded potential neurons and spiking neurons
            self.num_virtual_gpot_neurons = int(np.sum(num_input_gpot_neurons))
            self.num_virtual_spike_neurons = int(np.sum(num_input_spike_neurons))
            
            # cumulative sum of number of input neurons
            # the purpose is to indicate position in the buffer
            self.cum_virtual_gpot_neurons = np.concatenate(((0,), \
                np.cumsum(num_input_gpot_neurons))).astype(np.int32)
            self.cum_virtual_spike_neurons = np.concatenate(((0,), \
                np.cumsum(num_input_spike_neurons))).astype(np.int32)
            
            
        
        num_synapses = []
        cond_pre = []
        cond_post = []
        I_pre = []
        I_post = []
        reverse = []

        count = 0
        
        
        self.my_num_gpot_neurons = np.sum(self.num_gpot_neurons)
        self.my_num_spike_neurons = np.sum(self.num_spike_neurons)
        self.total_gpot_neurons = self.my_num_gpot_neurons + \
                                            self.num_virtual_gpot_neurons
        self.total_spike_neurons = self.my_num_spike_neurons + \
                                            self.num_virtual_spike_neurons

        
        for s_dict in s_dict_list:
            num_synapses.append(len(s_dict['type']))
            if s_dict['conductance'][0]:
                cond_post.extend(s_dict['post'])
                reverse.extend(s_dict['reverse'])
                cond_pre.extend(range(count, count+len(s_dict['post'])))
                count += len(s_dict['post'])
                if s_dict.has_key('delay'):
                    max_del = np.max(np.asarray(s_dict['delay']))
                    gpot_delay_steps = max_del if max_del > gpot_delay_steps \
                                       else gpot_delay_steps
            else:
                I_post.extend(s_dict['post'])
                I_pre.extend(range(count, count+len(s_dict['post'])))
                count += len(s_dict['post'])
                if s_dict.has_key('delay'):
                    max_del = np.max(np.asarray(s_dict['delay']))
                    spike_delay_steps = max_del if max_del > spike_delay_steps \
                                       else spike_delay_steps

        self.total_synapses = np.sum(num_synapses)
        I_post.extend(self.input_neuron_list)
        I_pre.extend(range(self.total_synapses, self.total_synapses + \
                          len(self.input_neuron_list)))

        
        cond_post = np.asarray(cond_post, dtype=np.int32)
        cond_pre = np.asarray(cond_pre, dtype = np.int32)
        reverse = np.asarray(reverse, dtype=np.double)

        order1 = np.argsort(cond_post)
        cond_post = cond_post[order1]
        cond_pre = cond_pre[order1]
        reverse = reverse[order1]

        
        I_post = np.asarray(I_post, dtype=np.int32)
        I_pre = np.asarray(I_pre, dtype=np.int32)
        
        order1 = np.argsort(I_post)
        I_post = I_post[order1]
        I_pre = I_pre[order1]

        self.idx_start_gpot = np.concatenate((np.asarray([0,], dtype=np.int32),\
                                np.cumsum(self.num_gpot_neurons, dtype=np.int32)))
        self.idx_start_spike = np.concatenate((np.asarray([0,], dtype=np.int32),\
                                np.cumsum(self.num_spike_neurons, dtype=np.int32)))
        self.idx_start_synapse = np.concatenate((np.asarray([0,], dtype=np.int32),\
                                        np.cumsum(num_synapses, dtype=np.int32)))

        
        for j, n_dict in enumerate(n_dict_list):
            if n_dict['spiking'][0]:
                # Note: The post id here is already translated. Don't need
                # to shift it again
                idx =  np.where(cond_post >= self.idx_start_spike[j])
                idx1 = np.where(cond_post < self.idx_start_spike[j+1])
                idx = np.intersect1d(idx, idx1, assume_unique=True)
                n_dict['cond_post'] = cond_post[idx] - self.idx_start_spike[j]
                n_dict['cond_pre'] = cond_pre[idx]
                n_dict['reverse'] = reverse[idx]
                idx =  np.where(I_post >= self.idx_start_spike[j])
                idx1 = np.where(I_post < self.idx_start_spike[j+1])
                idx = np.intersect1d(idx, idx1, assume_unique=True)
                n_dict['I_post'] = I_post[idx] - self.idx_start_spike[j]
                n_dict['I_pre'] = I_pre[idx]
            else:
                idx =  np.where(cond_post >= self.idx_start_gpot[j])
                idx1 = np.where(cond_post < self.idx_start_gpot[j+1])
                idx = np.intersect1d(idx, idx1, assume_unique=True)
                n_dict['cond_post'] = cond_post[idx] - self.idx_start_gpot[j]
                n_dict['cond_pre'] = cond_pre[idx]
                n_dict['reverse'] = reverse[idx]
                idx =  np.where(I_post >= self.idx_start_gpot[j])
                idx1 = np.where(I_post < self.idx_start_gpot[j+1])
                idx - np.intersect1d(idx, idx1, assume_unique=True)
                n_dict['I_post'] = I_post[idx] - self.idx_start_gpot[j]
                n_dict['I_pre'] = I_pre[idx]
                
            n_dict['num_dendrites_cond'] = Counter(n_dict['cond_post'])
            n_dict['num_dendrites_I'] = Counter(n_dict['I_post'])
        
        self.synapse_state = garray.zeros(self.total_synapses + \
                                    len(self.input_neuron_list), np.float64)
        if self.total_gpot_neurons>0:
            self.V = garray.zeros(self.total_gpot_neurons, np.float64)        

        if self.total_spike_neurons>0:
            self.spike_state = garray.zeros(self.total_spike_neurons, np.int32)

        self.n_dict_list = n_dict_list
        self.s_dict_list = s_dict_list

        self.input_neuron_list = np.asarray(self.input_neuron_list,
                                            dtype=np.int32)
        
        self.public_gpot_list.sort()
        self.public_gpot_list = np.asarray(self.public_gpot_list,dtype=np.int32)
        if len(self.public_gpot_list)>0:
            self.public_gpot_list_g = garray.to_gpu(self.public_gpot_list)
        self.public_spike_list.sort()
        self.public_spike_list = np.asarray(self.public_spike_list,
                                            dtype = np.int32)
        if len(self.public_spike_list)>0:
            self.public_spike_list_g = garray.to_gpu(self.public_spike_list)

        if len(self.public_gpot_list)>0:
            self.projection_gpot = garray.empty_like(self.public_gpot_list_g)

        if len(self.public_spike_list)>0:
            self.projection_spike = garray.empty_like(self.public_spike_list_g)

        self.num_public_gpot = len(self.public_gpot_list)
        self.num_public_spike = len(self.public_spike_list)
        self.num_input = len(self.input_neuron_list)
        
        self.gpot_delay_steps = int(round(gpot_delay_steps*1e-3 / self.dt)) + 1
        self.spike_delay_steps = int(round(spike_delay_steps*1e-3 / self.dt)) + 1
        

    def _read_LPU_input(self, in_gpot_dict, in_spike_dict):
        """
        Put inputs from other LPUs to buffer.
        
        """
    
        for i, gpot_data in enumerate(in_gpot_dict):
            if self.num_input_gpot_neurons[i] > 0:
                cuda.memcpy_htod(int(int(self.buffer.gpot_buffer.gpudata) \
                    +(self.buffer.gpot_current * self.buffer.gpot_buffer.ld \
                    + self.my_num_gpot_neurons + self.cum_virtual_gpot_neurons[i]) \
                    * self.buffer.gpot_buffer.dtype.itemsize), gpot_data)
            

        #Will need to change this if only spike indexes are transmitted
        for i, spike_data in range(in_spike_dict):
            if self.num_input_spike_neurons[i] > 0:
                cuda.memcpy_htod(int(int(self.buffer.spike_buffer.gpudata) \
                    +(self.buffer.spike_current * self.buffer.spike_buffer.ld \
                    + self.my_num_spike_neurons + self.cum_virtual_spike_neurons[i]) \
                    * self.buffer.spike_buffer.dtype.itemsize), spike_data)
            


    def _extract_output(self, out_gpot, out_spike, st=None):
        '''
        This function should be changed so that if the output attribute is True,
        the following kernel calls are not made as all the GPU data will have to be
        transferred to the CPU anyways.
        '''
        if self.num_public_gpot>0:
            self._extract_projection_gpot_func.prepared_async_call(\
                self.grid_extract_gpot, self.block_extract, st, self.V.gpudata, \
                self.projection_gpot.gpudata, self.public_gpot_list.gpudata, \
                self.num_public_gpot)
        if self.num_public_spike>0:
            self._extract_projection_gpot_func.prepared_async_call(\
                self.grid_extract_spike, self.block_extract, st, self.spike_state.gpudata, \
                self.projection_spike.gpudata, self.public_spike_list.gpudata, \
                self.num_public_gpot)

        if self.num_public_gpot>0:
            out_gpot.extend(self.projection_gpot.get())
        if self.num_public_spike>0:
            out_spike.extend(self.projection_spike.get())



    def _write_output(self):
        if self.my_num_gpot_neurons>0:
            self.output_gpot_file.root.array.append(self.V.get().reshape((1,-1)))
        if self.my_num_spike_neurons>0:
            self.output_spike_file.root.array.append(self.spike_state.get().reshape((1,-1)))
            


    def _read_external_input(self):
        if self.input_eof:
            return
        cuda.memcpy_dtod(int(int(self.synapse_state.gpudata) + \
            self.total_synapses*self.synapse_state.dtype.itemsize), \
            int(int(self.I_ext.gpudata) + self.frame_count*self.I_ext.ld*self.I_ext.dtype.itemsize), \
            self.num_input * self.synapse_state.dtype.itemsize)
        self.frame_count += 1
        if self.frame_count >= self.one_time_import:
            h_ext = self.input_h5file.root.real.read(self.file_pointer, self.file_pointer + self.one_time_import)
            if h_ext.shape[0] == self.I_ext.shape[0]:
                self.I_ext.set(h_ext)
                self.file_pointer += self.one_time_import
                self.frame_count = 0
            else:
                if self.file_pointer == self.input_h5file.root.real.shape[0]:
                    self.logger.info('Input end of file reached. Behaviour is' +\
                                    'undefined for subsequent steps')
                    self.input_eof = True

    
    def _update_buffer(self):
        if self.total_gpot_neurons>0:
            cuda.memcpy_dtod(int(self.buffer.gpot_buffer.gpudata) + \
                self.buffer.gpot_current*self.buffer.gpot_buffer.ld* \
                self.buffer.gpot_buffer.dtype.itemsize, self.V.gpudata, \
                int(self.V.dtype.itemsize*self.my_num_gpot_neurons))
        if self.total_spike_neurons>0:
            cuda.memcpy_dtod(int(self.buffer.spike_buffer.gpudata) + \
                self.buffer.spike_current*self.buffer.spike_buffer.ld* \
                self.buffer.spike_buffer.dtype.itemsize, self.spike_state.gpudata,\
                int(self.spike_state.dtype.itemsize*self.my_num_spike_neurons))
        self.buffer.step()



    def _extract_projection_gpot_func(self):
        template = """
        __global__ void extract_projection(%(type)s* all_V, %(type)s* projection_V, int* projection_list, int N)
        {
              int tid = threadIdx.x + blockIdx.x * blockDim.x;
              int total_threads = blockDim.x * gridDim.x;

              int ind;
              for(int i = tid; i < N; i += total_threads)
              {
                   ind = projection_list[i];
                   projection_V[i] = all_V[ind];
              }
        }

        """
        mod = SourceModule(template % {"type": dtype_to_ctype(self.V.dtype)}, options = ["--ptxas-options=-v"])
        func = mod.get_function("extract_projection")
        func.prepare([np.intp, np.intp, np.intp, np.int32])
        self.block_extract = (256,1,1)
        self.grid_extract_gpot = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                            (self.num_public_gpot-1) / 256 + 1), 1)
        return func



    def _extract_projection_spike_func(self):
        template = """
        __global__ void extract_projection(%(type)s* all_V, %(type)s* projection_V, int* projection_list, int N)
        {
              int tid = threadIdx.x + blockIdx.x * blockDim.x;
              int total_threads = blockDim.x * gridDim.x;

              int ind;
              for(int i = tid; i < N; i += total_threads)
              {
                   ind = projection_list[i];
                   projection_V[i] = all_V[ind];
              }
        }

        """
        mod = SourceModule(template % {"type": dtype_to_ctype(self.spike_state.dtype)}, options = ["--ptxas-options=-v"])
        func = mod.get_function("extract_projection")
        func.prepare([np.intp, np.intp, np.intp, np.int32])
        self.block_extract = (256,1,1)
        self.grid_extract_spike = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                            (self.num_public_spike-1) / 256 + 1), 1)
        return func
    


    def _instantiate_neuron(self, i):
        n_dict = self.n_dict_list[i]
        
        try:
            ind = self._neuron_names.index(n_dict['type'][0])
        except:
            self.logger.info('Error instantiating neurons of type' + \
                             n_dict['type'][0])
            return []
            
        if n_dict['spiking'][0]:
            neuron = self._neuron_classes[ind](n_dict, int(int(self.spike_state.gpudata) + \
                        self.spike_state.dtype.itemsize*self.idx_start_spike[i]), \
                        self.dt, debug=self.debug)
        else:
            neuron = self._neuron_classes[ind](n_dict, int(self.V.gpudata) +  \
                        self.V.dtype.itemsize*self.idx_start_gpot[i], \
                        self.dt, debug=self.debug)

        if not neuron.update_I_override:
            baseneuron.BaseNeuron.__init__(neuron, n_dict, int(int(self.V.gpudata) +  \
                        self.V.dtype.itemsize*self.idx_start_gpot[i]), \
                        self.dt, debug=self.debug)

        return neuron
        
    def _instantiate_synapse(self, i):
        s_dict = self.s_dict_list[i]
        try:
            ind = self._synapse_names.index(s_dict['type'][0])
        except:
            self.logger.info('Error instantiating synapses of type' + \
                             s_dict['type'][0])
            return []
        
        return self._synapse_classes[ind](s_dict, int(int(self.synapse_state.gpudata) + \
                self.synapse_state.dtype.itemsize*self.idx_start_synapse[i]), \
                self.dt, debug=self.debug)
    
        

    def _load_neurons(self):
        self._neuron_classes = baseneuron.BaseNeuron.__subclasses__()
        self._neuron_names = [cls.__name__ for cls in self._neuron_classes]

    def _load_synapses(self):
        self._synapse_classes = basesynapse.BaseSynapse.__subclasses__()
        self._synapse_names = [cls.__name__ for cls in self._synapse_classes]

class circular_array:
    '''
    This class implements a circular buffer to support synapses with delays.
    Please refer the documentation of the template synapse class on information
    on how to access data correctly from this buffer
    '''
    def __init__(self, num_gpot_neurons, gpot_delay_steps,
                 rest, num_spike_neurons, spike_delay_steps):

        self.num_gpot_neurons = num_gpot_neurons
        if num_gpot_neurons > 0:
            self.dtype = np.double
            self.gpot_delay_steps = gpot_delay_steps
            self.gpot_buffer = parray.empty((gpot_delay_steps, num_gpot_neurons),np.double)
            
            d_rest = garray.to_gpu(rest)
            self.gpot_current = 0

            for i in range(gpot_delay_steps):
                cuda.memcpy_dtod(int(self.gpot_buffer.gpudata) + \
                    self.gpot_buffer.ld * i * self.gpot_buffer.dtype.itemsize,\
                    d_rest.gpudata, d_rest.nbytes)
                
            self.num_spike_neurons = num_spike_neurons
                
        if num_spike_neurons > 0:
            self.spike_delay_steps = spike_delay_steps
            self.spike_buffer = parray.zeros((spike_delay_steps,num_spike_neurons),np.int32)
            self.spike_current = 0

    def step(self):
        if self.num_gpot_neurons > 0:
            self.gpot_current += 1
            if self.gpot_current >= self.gpot_delay_steps:
                self.gpot_current = 0

        if self.num_spike_neurons > 0:
            self.spike_current += 1
            if self.spike_current >= self.spike_delay_steps:
                self.spike_current = 0

    def update_other_rest(self, other_rest, my_num_gpot_neurons, num_virtual_gpot_neurons):
        if self.num_gpot_neurons > 0:
            d_other_rest = garray.zeros(num_virtual_gpot_neurons, np.double)
            a = 0
            for i in range(len(other_rest)):
                if other_rest[i].size > 0:
                    cuda.memcpy_dtod(int(d_other_rest.gpudata) +  a ,
                                     other_rest[i].gpudata, other_rest[i].nbytes)
                    a += other_rest[i].nbytes

            for i in range(self.gpot_delay_steps):
                cuda.memcpy_dtod( int(self.gpot_buffer.gpudata) + \
                    (self.gpot_buffer.ld * i + my_num_gpot_neurons) * \
                    self.gpot_buffer.dtype.itemsize, d_other_rest.gpudata, \
                    d_other_rest.nbytes )
