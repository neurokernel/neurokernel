#!/usr/bin/env python

"""
Local Processing Unit (LPU) draft implementation.
"""

import collections

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np
import networkx as nx

# Work around bug that causes networkx to choke on GEXF files with boolean
# attributes that contain the strings 'True' or 'False'
# (bug already observed in https://github.com/networkx/networkx/pull/971)
nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                        'true':True, 'True':True}

from neurokernel.core import Module
import neurokernel.base as base

from types import *
from collections import Counter

from utils.simpleio import *
import utils.parray as parray
from neurons import *
from synapses import *

import pdb

class LPU(Module, object):
    """
    Local Processing Unit (LPU).

    Parameters
    ----------
    dt : double
        Time step (s).
    n_dict_list : list of dict
        List of dictionaries describing the neurons in this LPU; each dictionary
        corresponds to a single neuron model.
    s_dict_list : list of dict
        List of dictionaries describing the synapses in this LPU; each
        dictionary corresponds to a single synapse model.
    input_file : str
        Name of input file
    output_file : str
        Name of output files
    port_data : int
        Port to use when communicating with broker.
    port_ctrl : int
        Port used by broker to control module.
    device : int
        GPU device number.
    id : str
        Name of the LPU
    debug : boolean
        Passed to all the neuron and synapse objects instantiated by this LPU
        for debugging purposes. False by default.
    """

    @staticmethod
    def lpu_parser(filename):
        """
        GEXF LPU specification parser.

        Extract LPU specification data from a GEXF file and store it
        in a list of dictionaries. All nodes in the GEXF file are assumed to
        correspond to neuron model instances while all edges are assumed to
        correspond to synapse model instances.

        Parameters
        ----------
        filename : str
            GEXF filename.

        Returns
        -------
        n_dict : dict of dict of neuron
            Each key of `n_dict` is the name of a neuron model; the values
            are dicts that map each attribute name to a list that contains the
            attribute values for each neuron.
        s_dict : dict of dict of synapse
            Each key of `s_dict` is the name of a synapse model; the values are
            dicts that map each attribute name to a list that contains the
            attribute values for each each neuron.

        Example
        -------
        >>> n_dict = {'LeakyIAF': {'Vr': [0.5, 0.6], 'Vt': [0.3, 0.2]},
                      'MorrisLecar': {'V1': [0.15, 0.16], 'Vt': [0.13, 0.27]}}

        Notes
        -----
        All neurons must have the following attributes; any additional attributes
        for a specific neuron model must be provided for all neurons of that
        model type:

        1. spiking - True if the neuron emits spikes, False if it emits graded
           potentials.
        2. model - model identifier string, e.g., 'LeakyIAF', 'MorrisLecar'
        3. public - True if the neuron emits output exposed to other LPUS.
        4. extern - True if the neuron can receive external input from a file.

        All synapses must have the following attributes:

        1. class - int indicating connection class of synapse; it may assume the
           following values:

            0. spike to spike synapse
            1. spike to graded potential synapse
            2. graded potential to spike synapse
            3. graded potential to graded potential synapse
        2. model - model identifier string, e.g., 'AlphaSynapse'
        3. conductance - True if the synapse emits conductance values, False if
           it emits current values.
        4. reverse - If the `conductance` attribute is True, this attribute
           should be set to the reverse potential.

        TODO
        ----
        Input data should be validated.
        """

        # parse the GEXF file using networkX
        graph = nx.read_gexf(filename)

        # parse neuron data
        n_dict = {}
        neurons = graph.node.items()
        neurons.sort( cmp=neuron_cmp  )
        for id, neu in neurons:
            model = neu['model']
            # if an input_port, make sure selector is specified
            if model == 'port_in_gpot' or model == 'port_in_spk':
                assert('selector' in neu.keys())
                if model == 'port_in_gpot':
                    neu['spiking'] = False
                    neu['public'] = False
                else:
                    neu['spiking'] = True
                    neu['public'] = False
            # if an output_port, make sure selector is specified
            if 'public' in neu.keys():
                if neu['public']:
                    assert('selector' in neu.keys())
            else:
                neu['public'] = False
            if 'selector' not in neu.keys():
                neu['selector'] = ''
            # if the neuron model does not appear before, add it into n_dict
            if model not in n_dict:
                n_dict[model] = { k:[] for k in neu.keys()+['id'] }
            # add neuron data into the subdictionary of n_dict
            for key in neu.iterkeys():
                n_dict[model][key].append( neu[key] )
            n_dict[model]['id'].append( int(id) )
        # remove duplicate model information
        for val in n_dict.itervalues(): val.pop('model')
        if not n_dict: n_dict = None

        # parse synapse data
        synapses = graph.edges(data=True)
        s_dict = {}
        synapses.sort(cmp=synapse_cmp)
        for syn in synapses:
            # syn[0/1]: pre-/post-neu id; syn[2]: dict of synaptic data
            model = syn[2]['model']
            syn[2]['id'] = int( syn[2]['id'] )
            # if the sysnapse model does not apear before, add it into s_dict
            if model not in s_dict:
                s_dict[model] = { k:[] for k in syn[2].keys()+['pre','post'] }
            # add sysnaptic data into the subdictionary of s_dict
            for key in syn[2].iterkeys():
                s_dict[model][key].append( syn[2][key] )
            s_dict[model]['pre'].append( syn[0] )
            s_dict[model]['post'].append( syn[1] )
        for val in s_dict.itervalues(): val.pop('model')
        if not s_dict: s_dict = {}
        return n_dict, s_dict

    def __init__(self, dt, n_dict, s_dict, input_file=None, device=0,
                 output_file=None,port_ctrl=base.PORT_CTRL,port_data=base.PORT_DATA,
                 id=None, debug=False, columns = [ 'io', 'type']):
        assert('io' in columns)
        assert('type' in columns)
        columns.append('interface')
        self.LPU_id = id
        self.dt = dt
        self.debug = debug
        self.device = device

        # handle file I/O
        self.output_file = output_file
        self.output = True if output_file else False
        self.input_file = input_file
        self.input_eof = False if input_file else True

        # load neurons and synapse definition
        self._load_neurons()
        self._load_synapses()

        # set default one time import
        self._one_time_import = 10
        #TODO: comment
        self.n_list = n_dict.items()
        n_model_is_spk = [ n['spiking'][0] for t,n in self.n_list ]
        n_model_num = [ len(n['id']) for t,n in self.n_list ]
        n_id = np.array(sum( [ n['id'] for t,n in self.n_list ], []), dtype=np.int32)
        n_is_spk = np.array(sum( [ n['spiking'] for t,n in self.n_list ], []))
        n_is_pub = np.array(sum( [ n['public'] for t,n in self.n_list ], []))
        n_has_in = np.array(sum( [ n['extern'] for t,n in self.n_list ], []))

        
        if 'port_in_gpot' in n_dict.keys() or 'port_in_spk' in n_dict.keys():
            sel_in_gpot = ','.join( [sel for sel in n_dict['port_in_gpot']['selector'] ])
            sel_in_spk = ','.join( [sel for sel in n_dict['port_in_spk']['selector'] ])
            if sel_in_gpot[-1] == ',' : sel_in_gpot = sel_in_gpot[:-1]
            if sel_in_gpot and sel_in_gpot[0] == ',': sel_in_gpot = sel_in_gpot[1:]
            if sel_in_spk[-1] == ',' : sel_in_spk = sel_in_spk[:-1]
            if sel_in_spk and sel_in_spk[0] == ',': sel_in_spk = sel_in_spk[1:]
            self.in_ports_ids_gpot = np.array([id for id in n_dict['port_in_gpot']['id'] ])
            self.in_ports_ids_spk = np.array([id for id in n_dict['port_in_spk']['id'] ])
            self.ports_in_gpot_mem_ind = zip(*self.n_list)[0].index('port_in_gpot')
            self.ports_in_spk_mem_ind = zip(*self.n_list)[0].index('port_in_spk')
        else:
            sel_in_gpot = ''
            sel_in_spk = ''
            self.in_ports_ids_spk = np.array( [] )
            self.in_ports_ids_gpot = np.array( [] )
            self.ports_in_gpot_mem_ind = None
            self.ports_in_spk_mem_ind = None
            
        sel_in = ','.join([sel_in_gpot, sel_in_spk])
        if sel_in[-1] == ',' : sel_in = sel_in[:-1]
        if sel_in and sel_in[0] == ',': sel_in = sel_in[1:]
        
        sel_out_gpot = ','.join( [ sel for t,n in self.n_list for sel,pub,spk in
                                   zip(n['selector'],n['public'],n['spiking'])
                                   if pub and not spk ] )
        sel_out_spk = ','.join( [ sel for t,n in self.n_list for sel,pub,spk in
                                   zip(n['selector'],n['public'],n['spiking'])
                                   if pub and spk ] )
        self.out_ports_ids_gpot = np.array( [ id for t,n in self.n_list for id, pub, spk in
                                              zip(n['id'],n['public'],n['spiking'])
                                              if pub and not spk ] )
        self.out_ports_ids_spk = np.array( [ id for t,n in self.n_list for id, pub, spk in
                                             zip(n['id'],n['public'],n['spiking'])
                                             if pub and spk ] )
        if sel_out_gpot:
            if sel_out_gpot[-1] == ',' : sel_out_gpot = sel_out_gpot[:-1]
            if sel_out_gpot and sel_out_gpot[0] == ',': sel_out_gpot = sel_out_gpot[1:]
        if sel_out_spk:
            if sel_out_spk[-1] == ',' : sel_out_spk = sel_out_spk[:-1]
            if sel_out_spk and sel_out_spk[0] == ',': sel_out_spk = sel_out_spk[1:]
        
        sel_out = ','.join([sel_out_gpot, sel_out_spk])
        if sel_out:
            if sel_out[-1] == ',' : sel_out = sel_out[:-1]
            if sel_out and sel_out[0] == ',': sel_out = sel_out[1:]

        sel_gpot = ','.join([sel_in_gpot, sel_out_gpot])
        sel_spk = ','.join([sel_in_spk, sel_out_spk])
        if sel_gpot:
            if sel_gpot[-1] == ',' : sel_gpot = sel_gpot[:-1]
            if sel_gpot and sel_gpot[0] == ',': sel_gpot = sel_gpot[1:]
        if sel_spk:
            if sel_spk[-1] == ',' : sel_spk = sel_spk[:-1]
            if sel_spk and sel_spk[0] == ',': sel_spk = sel_spk[1:]

        sel = ','.join( [sel_gpot,sel_spk] )
        if sel:
            if sel[-1] == ',' : sel = sel[:-1]
            if sel and sel[0] == ',': sel = sel[1:]

        self.sel_in_spk = sel_in_spk
        self.sel_out_spk = sel_out_spk
        self.sel_in_gpot = sel_in_gpot
        self.sel_out_gpot = sel_out_gpot
        
        #TODO: comment
        self.num_gpot_neurons = np.where( n_model_is_spk, 0, n_model_num)
        self.num_spike_neurons = np.where( n_model_is_spk, n_model_num, 0)
        self.my_num_gpot_neurons = sum( self.num_gpot_neurons )
        self.my_num_spike_neurons = sum( self.num_spike_neurons )
        self.gpot_idx = n_id[ ~n_is_spk ]
        self.spike_idx = n_id[ n_is_spk ]
        self.order = np.argsort( np.concatenate((self.gpot_idx, self.spike_idx ))).astype(np.int32)
        self.gpot_order = np.argsort( self.gpot_idx ).astype(np.int32)
        self.spike_order = np.argsort( self.spike_idx ).astype(np.int32)
        self.spike_shift = self.my_num_gpot_neurons
        in_id = n_id[n_has_in]
        in_id.sort()
        pub_spk_id = n_id[ n_is_pub & n_is_spk ]
        pub_spk_id.sort()
        pub_gpot_id = n_id[ n_is_pub & ~n_is_spk ]
        pub_gpot_id.sort()
        self.input_neuron_list = self.order[in_id]
        self.public_spike_list = self.order[pub_spk_id]
        self.public_gpot_list = self.order[pub_gpot_id]
        self.num_public_gpot = len( self.public_gpot_list )
        self.num_public_spike = len( self.public_spike_list )
        self.num_input = len( self.input_neuron_list )
        if len(self.in_ports_ids_gpot)>0:
            self.in_ports_ids_gpot = self.order[self.in_ports_ids_gpot]
        if len(self.in_ports_ids_spk)>0:
            self.in_ports_ids_spk = self.order[self.in_ports_ids_spk]
        if len(self.out_ports_ids_gpot)>0:
            self.out_ports_ids_gpot = self.order[self.out_ports_ids_gpot]
        if len(self.out_ports_ids_spk)>0:
            self.out_ports_ids_spk = self.order[self.out_ports_ids_spk]
        
        #TODO: comment
        self.s_dict = s_dict
        if s_dict:
            for s in self.s_dict.itervalues():
                shift = self.spike_shift if s['class'][0] == 0 or s['class'][0] == 1 else 0
                s['pre'] = [ self.order[int(neu_id)]-shift for neu_id in s['pre'] ]
                s['post'] = [ self.order[int(neu_id)] for neu_id in s['post'] ]
                
        gpot_delay_steps = 0
        spike_delay_steps = 0


        order = self.order
        spike_shift = self.spike_shift

        cond_pre = []
        cond_post = []
        I_pre = []
        I_post = []
        reverse = []

        count = 0

        self.s_list = self.s_dict.items()
        num_synapses = [ len(s['id']) for t,s in self.s_list ]
        for (t,s) in self.s_list:
            order = np.argsort(s['post']).astype(np.int32)
            for k,v in s.items():
                s[k] = np.asarray(v)[order]
            
            if s['conductance'][0]:
                cond_post.extend(s['post'])
                reverse.extend(s['reverse'])
                cond_pre.extend(range(count, count+len(s['post'])))
                count += len(s['post'])
                if 'delay' in s:
                    max_del = np.max( s['delay'] )
                    gpot_delay_steps = max_del if max_del > gpot_delay_steps \
                                       else gpot_delay_steps
            else:
                I_post.extend(s['post'])
                I_pre.extend(range(count, count+len(s['post'])))
                count += len(s['post'])
                if 'delay' in s:
                    max_del = np.max( s['delay'] )
                    spike_delay_steps = max_del if max_del > spike_delay_steps \
                                       else spike_delay_steps
        
        self.total_synapses = int(np.sum(num_synapses))
        I_post.extend(self.input_neuron_list)
        I_pre.extend(range(self.total_synapses, self.total_synapses + \
                          len(self.input_neuron_list)))


        cond_post = np.asarray(cond_post, dtype=np.int32)
        cond_pre = np.asarray(cond_pre, dtype = np.int32)
        reverse = np.asarray(reverse, dtype=np.double)

        order1 = np.argsort(cond_post, kind='mergesort')
        cond_post = cond_post[order1]
        cond_pre = cond_pre[order1]
        reverse = reverse[order1]


        I_post = np.asarray(I_post, dtype=np.int32)
        I_pre = np.asarray(I_pre, dtype=np.int32)

        order1 = np.argsort(I_post, kind='mergesort')
        I_post = I_post[order1]
        I_pre = I_pre[order1]

        self.idx_start_gpot = np.concatenate((np.asarray([0,], dtype=np.int32),\
                                np.cumsum(self.num_gpot_neurons, dtype=np.int32)))
        self.idx_start_spike = np.concatenate((np.asarray([0,], dtype=np.int32),\
                                np.cumsum(self.num_spike_neurons, dtype=np.int32)))
        self.idx_start_synapse = np.concatenate((np.asarray([0,], dtype=np.int32),\
                                        np.cumsum(num_synapses, dtype=np.int32)))


        for i,(t,n) in enumerate(self.n_list):
            if n['spiking'][0]:
                idx = np.where( (cond_post >= (self.idx_start_spike[i]+self.spike_shift)) \
                               &(cond_post < (self.idx_start_spike[i+1]+self.spike_shift)) )
                n['cond_post'] = cond_post[idx] - self.idx_start_spike[i] - self.spike_shift
                n['cond_pre'] = cond_pre[idx]
                n['reverse'] = reverse[idx]
                idx = np.where( (I_post >= self.idx_start_spike[i]+self.spike_shift) \
                               &(I_post < self.idx_start_spike[i+1]+self.spike_shift) )
                n['I_post'] = I_post[idx] - self.idx_start_spike[i] - spike_shift
                n['I_pre'] = I_pre[idx]
            else:
                idx = np.where( (cond_post >= self.idx_start_gpot[i]) \
                               &(cond_post < self.idx_start_gpot[i+1]) )
                n['cond_post'] = cond_post[idx] - self.idx_start_gpot[i]
                n['cond_pre'] = cond_pre[idx]
                n['reverse'] = reverse[idx]
                idx =  np.where( (I_post >= self.idx_start_gpot[i]) \
                                &(I_post < self.idx_start_gpot[i+1]) )
                n['I_post'] = I_post[idx] - self.idx_start_gpot[i]
                n['I_pre'] = I_pre[idx]

            n['num_dendrites_cond'] = Counter(n['cond_post'])
            n['num_dendrites_I'] = Counter(n['I_post'])

        self.gpot_delay_steps = int(round(gpot_delay_steps*1e-3 / self.dt)) + 1
        self.spike_delay_steps = int(round(spike_delay_steps*1e-3 / self.dt)) + 1


        data_gpot = np.zeros(self.num_public_gpot, np.double)
        data_spike = np.zeros(self.num_public_spike, np.bool)
        super(LPU, self).__init__(sel, sel_gpot, sel_spk, data_gpot, data_spike,
                                  columns, port_data, port_ctrl, self.LPU_id, device, debug)

        self.interface[sel_in_gpot, 'io', 'type'] = ['in', 'gpot']
        self.interface[sel_out_gpot, 'io', 'type'] = ['out', 'gpot']
        self.interface[sel_in_spk, 'io', 'type'] = ['in', 'spike']
        self.interface[sel_out_spk, 'io', 'type'] = ['out', 'spike']
        

    
    def pre_run(self):
        super(LPU, self).pre_run()
        self._initialize_gpu_ds()
        self._init_objects()
        
        
    def post_run(self):
        super(LPU, self).post_run()
        if self.output:
            if self.my_num_gpot_neurons > 0:
                self.output_gpot_file.close()
            if self.my_num_spike_neurons > 0:
                self.output_spike_file.close()
        if self.debug:
            # for file in self.in_gpot_files.itervalues():
            #     file.close()
            self.gpot_buffer_file.close()
            self.synapse_state_file.close()
        for neuron in self.neurons:
            neuron.post_run()
            if self.debug and not neuron.update_I_override:
                neuron._BaseNeuron__post_run()
                
        for synapse in self.synapses:
            synapse.post_run()



    def run_step(self):
        super(LPU, self).run_step()

        self._read_LPU_input()
        
        
        if self.input_file is not None:
            self._read_external_input()

        for neuron in self.neurons:
            neuron.update_I(self.synapse_state.gpudata)
            neuron.eval()

        self._update_buffer()
        for synapse in self.synapses:
            synapse.update_state(self.buffer)

        if self.debug:
            self.gpot_buffer_file.root.array.append( \
                self.buffer.gpot_buffer.get().reshape(1,self.gpot_delay_steps,-1) )
            self.synapse_state_file.root.array.append( \
                self.synapse_state.get().reshape(1,-1))
        
        self.buffer.step()

        self._extract_output()

        # Save output data to disk:
        if self.output:
            self._write_output()

        
    def _init_objects(self):
        self.neurons = [ self._instantiate_neuron(i,t,n) for i,(t,n) in enumerate(self.n_list) \
                         if t != 'port_in_got' and t!='port_in_spk']
        self.synapses = [ self._instantiate_synapse(i,t,n) for i,(t,n) in enumerate(self.s_list)
                          if t != 'pass']
        self.buffer = circular_array(self.my_num_gpot_neurons,
                                     self.gpot_delay_steps, self.V, 
                                     self.my_num_spike_neurons, self.spike_delay_steps)
        if self.input_file:
            self.input_h5file = tables.openFile(self.input_file)

            self.file_pointer = 0
            self.I_ext = parray.to_gpu(self.input_h5file.root.array.read(\
                                self.file_pointer, self.file_pointer + \
                                self._one_time_import))
            self.file_pointer += self._one_time_import
            self.frame_count = 0
            self.frames_in_buffer = self._one_time_import

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

        if self.debug:
            '''
            self.in_gpot_files = {}
            for (key,i) in self.other_lpu_map.iteritems():
                num = self.num_input_gpot_neurons[i]
                if num>0:
                    self.in_gpot_files[key] = tables.openFile(filename + \
                                                    key + '_in_gpot.' + ext , 'w')
                    self.in_gpot_files[key].createEArray("/","array", \
                                                        tables.Float64Atom(), (0,num))

            '''
            self.gpot_buffer_file = tables.openFile(self.id + '_buffer.h5','w')
            self.gpot_buffer_file.createEArray("/","array", \
                        tables.Float64Atom(), (0,self.gpot_delay_steps, self.my_num_gpot_neurons))

            self.synapse_state_file = tables.openFile(self.id + '_synapses.h5','w')
            self.synapse_state_file.createEArray("/","array", \
                        tables.Float64Atom(), (0, self.total_synapses + len(self.input_neuron_list)))

    def _initialize_gpu_ds(self):
        """
        Setup GPU arrays.
        """
        
        self.synapse_state = garray.zeros(int(self.total_synapses) + \
                                    len(self.input_neuron_list), np.float64)
        
        if self.my_num_gpot_neurons>0:
            self.V = garray.zeros(int(self.my_num_gpot_neurons), np.float64)
        else:
            self.V = None

        if self.my_num_spike_neurons>0:
            self.spike_state = garray.zeros(int(self.my_num_spike_neurons), np.int32)

        if len(self.out_ports_ids_gpot)>0:
            self.out_ports_ids_gpot_g = garray.to_gpu(self.out_ports_ids_gpot)
            self.out_port_data_gpot = garray.zeros(len(self.out_ports_ids_gpot), np.double)
            self._extract_gpot = self._extract_projection_gpot_func()

        if len(self.out_ports_ids_spk)>0:
            self.out_ports_ids_spk_g = garray.to_gpu( \
                (self.out_ports_ids_spk-self.spike_shift).astype(np.int32))
            self.out_port_data_spk = garray.zeros(len(self.out_ports_ids_spk), np.int32)
            self._extract_spike = self._extract_projection_spike_func()




    def _read_LPU_input(self):
        """
        Put inputs from other LPUs to buffer.

        """
        
        if self.ports_in_gpot_mem_ind:
            cuda.memcpy_htod(int(self.V.gpudata) + \
            self.V.dtype.itemsize*self.idx_start_gpot[self.ports_in_gpot_mem_ind],
            self.pm['gpot'][self.sel_in_gpot])

        if self.ports_in_spk_mem_ind:
            cuda.memcpy_htod(int(int(self.spike_state.gpudata) + \
            self.spike_state.dtype.itemsize*self.idx_start_spike[self.ports_in_spk_mem_ind], \
            self.pm['spike'][self.sel_in_spk]))


    def _extract_output(self, st=None):
        """
        This function should be changed so that if the output attribute is True,
        the following kernel calls are not made as all the GPU data will have to be
        transferred to the host anyways.
        """
        if len(self.out_ports_ids_gpot)>0:
            self._extract_gpot.prepared_async_call(\
                self.grid_extract_gpot, self.block_extract, st, self.V.gpudata, \
                self.out_port_data_gpot.gpudata, self.out_ports_ids_gpot_g.gpudata, \
                self.num_public_gpot)
        if len(self.out_ports_ids_spk)>0:
            self._extract_spike.prepared_async_call(\
                self.grid_extract_spike, self.block_extract, st, self.spike_state.gpudata, \
                self.out_port_data_spk.gpudata, self.out_ports_ids_spk_g.gpudata, \
                len(self.out_ports_ids_spk))

        # Save the states of the graded potential neurons and the indices of the
        # spiking neurons that have emitted a spike:
        if len(self.out_ports_ids_gpot)>0:
            self.pm['gpot'][self.sel_out_gpot] = (self.out_port_data_gpot.get())
        if len(self.out_ports_ids_spk)>0:
            self.pm['spike'][self.sel_out_spk] = (self.out_port_data_spk.get())

    def _write_output(self):
        """
        Save neuron states or spikes to output file.
        """

        if self.my_num_gpot_neurons>0:
            self.output_gpot_file.root.array.append(self.V.get()\
                [self.gpot_order].reshape((1,-1)))
        if self.my_num_spike_neurons>0:
            self.output_spike_file.root.array.append(self.spike_state.get()\
                [self.spike_order].reshape((1,-1)))

    def _read_external_input(self):
        if not self.input_eof or self.frame_count<self.frames_in_buffer:
            cuda.memcpy_dtod(int(int(self.synapse_state.gpudata) + \
                             self.total_synapses*self.synapse_state.dtype.itemsize), \
                             int(int(self.I_ext.gpudata) + self.frame_count*self.I_ext.ld*self.I_ext.dtype.itemsize), \
                             self.num_input * self.synapse_state.dtype.itemsize)
            self.frame_count += 1
        else:
            self.logger.info('Input end of file reached. Subsequent behaviour is undefined.')
        if self.frame_count >= self._one_time_import and not self.input_eof:
            input_ld = self.input_h5file.root.array.shape[0]
            if input_ld - self.file_pointer < self._one_time_import:
                h_ext = self.input_h5file.root.array.read(self.file_pointer, input_ld)
            else:
                h_ext = self.input_h5file.root.array.read(self.file_pointer, self.file_pointer + self._one_time_import)
            if h_ext.shape[0] == self.I_ext.shape[0]:
                self.I_ext.set(h_ext)
                self.file_pointer += self._one_time_import
                self.frame_count = 0
            else:
                pad_shape = list(h_ext.shape)
                self.frames_in_buffer = h_ext.shape[0]
                pad_shape[0] = self._one_time_import - h_ext.shape[0]
                h_ext = np.concatenate((h_ext, np.zeros(pad_shape)), axis=0)
                self.I_ext.set(h_ext)
                self.file_pointer = input_ld
                
            if self.file_pointer == self.input_h5file.root.array.shape[0]:
                self.input_eof = True
                    

    #TODO
    def _update_buffer(self):
        if self.my_num_gpot_neurons>0:
            cuda.memcpy_dtod(int(self.buffer.gpot_buffer.gpudata) + \
                self.buffer.gpot_current*self.buffer.gpot_buffer.ld* \
                self.buffer.gpot_buffer.dtype.itemsize, self.V.gpudata, \
                self.V.nbytes)
        if self.my_num_spike_neurons>0:
            cuda.memcpy_dtod(int(self.buffer.spike_buffer.gpudata) + \
                self.buffer.spike_current*self.buffer.spike_buffer.ld* \
                self.buffer.spike_buffer.dtype.itemsize,
                self.spike_state.gpudata,\
                int(self.spike_state.dtype.itemsize*self.my_num_spike_neurons))


    #TODO
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


    #TODO
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


    #TODO
    def _instantiate_neuron(self, i, t, n):
        try:
            ind = self._neuron_names.index(t)
            #ind = int(t)
        except:
            try:
                ind = int(t)
            except:
                self.logger.info('Error instantiating neurons of model \'%s\'' % t)
                return []

        if n['spiking'][0]:
            neuron = self._neuron_classes[ind]( n, int(int(self.spike_state.gpudata) + \
                        self.spike_state.dtype.itemsize*self.idx_start_spike[i]), \
                                                self.dt, debug=self.debug, LPU_id=self.id)
        else:
            neuron = self._neuron_classes[ind](n, int(self.V.gpudata) +  \
                        self.V.dtype.itemsize*self.idx_start_gpot[i], \
                                               self.dt, debug=self.debug)

        if not neuron.update_I_override:
            baseneuron.BaseNeuron.__init__(neuron, n, int(int(self.V.gpudata) +  \
                        self.V.dtype.itemsize*self.idx_start_gpot[i]), \
                                           self.dt, debug=self.debug, LPU_id=self.id)

        return neuron

    #TODO
    def _instantiate_synapse(self, i, t, s):
        try:
            ind = self._synapse_names.index( t )
            # ind = int(t)
        except:
            try:
                ind = int(t)
            except:
                self.logger.info('Error instantiating synapses of model \'%s\'' % t)
                return []

        return self._synapse_classes[ind](s, int(int(self.synapse_state.gpudata) + \
                self.synapse_state.dtype.itemsize*self.idx_start_synapse[i]), \
                self.dt, debug=self.debug)


    #TODO
    def _load_neurons(self):
        self._neuron_classes = baseneuron.BaseNeuron.__subclasses__()
        self._neuron_names = [cls.__name__ for cls in self._neuron_classes]

    #TODO
    def _load_synapses(self):
        self._synapse_classes = basesynapse.BaseSynapse.__subclasses__()
        self._synapse_names = [cls.__name__ for cls in self._synapse_classes]

    @property
    def one_time_import(self):
        return self._one_time_import

    @one_time_import.setter
    def one_time_import(self, value):
        self._one_time_import = value

def neuron_cmp( x, y):
    if int(x[0]) < int(y[0]):
        return -1
    elif int(x[0]) > int(y[0]):
        return 1
    else:
        return 0

def synapse_cmp( x, y):
    if int(x[1]) < int(y[1]):
        return -1
    elif int(x[1]) > int(y[1]):
        return 1
    else:
        return 0


class circular_array:
    '''
    This class implements a circular buffer to support synapses with delays.
    Please refer the documentation of the template synapse class on information
    on how to access data correctly from this buffer
    '''
    def __init__(self, num_gpot_neurons,  gpot_delay_steps,
                 rest, num_spike_neurons, spike_delay_steps):

        self.num_gpot_neurons = num_gpot_neurons
        if num_gpot_neurons > 0:
            self.dtype = np.double
            self.gpot_delay_steps = gpot_delay_steps
            self.gpot_buffer = parray.empty((gpot_delay_steps, num_gpot_neurons),np.double)

            self.gpot_current = 0
            
            for i in range(gpot_delay_steps):
                cuda.memcpy_dtod(int(self.gpot_buffer.gpudata) + \
                                 self.gpot_buffer.ld * i * self.gpot_buffer.dtype.itemsize,\
                                 rest.gpudata, rest.nbytes)

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

    '''
    def update_other_rest(self, gpot_data, my_num_gpot_neurons, num_virtual_gpot_neurons):
        # ??? is the second part of the below condition correct?
        if num_virtual_gpot_neurons > 0:            
            d_other_rest = garray.zeros(num_virtual_gpot_neurons, np.double)
            a = 0
            for data in gpot_data.itervalues():
                if len(data) > 0:
                    cuda.memcpy_htod(int(d_other_rest.gpudata) +  a , data)
                    a += data.nbytes

            for i in range(self.gpot_delay_steps):
                cuda.memcpy_dtod( int(self.gpot_buffer.gpudata) + \
                    (self.gpot_buffer.ld * i + int(my_num_gpot_neurons)) * \
                    self.gpot_buffer.dtype.itemsize, d_other_rest.gpudata, \
                    d_other_rest.nbytes )
    '''
