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
            # if the neuron model does not appear before, add it into n_dict
            if model not in n_dict:
                n_dict[model] = { k:[] for k in neu.keys()+['id'] }
            # add neuron data into the subdictionary of n_dict
            for key in neu.iterkeys():
                n_dict[model][key].append( neu[key] )
            n_dict[model]['id'].append( id ) # id is a string in GXEF
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
                 output_file=None, port_ctrl=base.PORT_CTRL,
                 port_data=base.PORT_DATA, id=None, debug=False):
        super(LPU, self).__init__(port_data=port_data, port_ctrl=port_ctrl,
                                  device=device, id=id)

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

        #TODO: comment
        self.s_dict = s_dict
        if s_dict:
            for s in self.s_dict.itervalues():
                shift = self.spike_shift if s['class'][0] == 0 or s['class'][0] == 1 else 0
                s['pre'] = [ self.order[int(neu_id)]-shift for neu_id in s['pre'] ]
                s['post'] = [ self.order[int(neu_id)] for neu_id in s['post'] ]

    def compare( self, lpu ):
        """
        Testing purpose
        """
        attrs = [ 'num_gpot_neurons', 'num_spike_neurons',
                  'my_num_gpot_neurons', 'my_num_spike_neurons',
                  'gpot_idx','spike_idx','order','spike_shift',
                  'input_neuron_list','num_input',
                  'public_spike_list','num_public_spike',
                  'public_gpot_list','num_public_gpot']
        all_the_same = True
        for attr in attrs:
            print "Comparing %s..." % attr,
            c = getattr(self,attr) == getattr(lpu,attr)
            if isinstance( c, collections.Iterable):
                c = all( np.asarray(c) )
            print "same!" if c else "different!!"
            all_the_same &= c
        if all_the_same: print "All attributes are the same"

    @property
    def N_gpot(self): return self.num_public_gpot

    @property
    def N_spike(self): return self.num_public_spike

    def pre_run(self):
        super(LPU, self).pre_run()
        self._setup_connectivity()
        self._initialize_gpu_ds()
        self._init_objects()
        self.buffer = circular_array(self.total_gpot_neurons, self.my_num_gpot_neurons,\
                        self.gpot_delay_steps, self.V, \
                        self.total_spike_neurons, self.spike_delay_steps)

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
            self.in_gpot_files = {}
            for (key,i) in self.other_lpu_map.iteritems():
                num = self.num_input_gpot_neurons[i]
                if num>0:
                    self.in_gpot_files[key] = tables.openFile(filename + \
                                                    key + '_in_gpot.' + ext , 'w')
                    self.in_gpot_files[key].createEArray("/","array", \
                                                        tables.Float64Atom(), (0,num))

            self.gpot_buffer_file = tables.openFile(self.id + '_buffer.h5','w')
            self.gpot_buffer_file.createEArray("/","array", \
                                               tables.Float64Atom(), (0,self.gpot_delay_steps, self.total_gpot_neurons))

    def post_run(self):
        super(LPU, self).post_run()
        if self.output:
            if self.my_num_gpot_neurons > 0:
                self.output_gpot_file.close()
            if self.my_num_spike_neurons > 0:
                self.output_spike_file.close()
        if self.debug:
            for file in self.in_gpot_files.itervalues():
                file.close()
            self.gpot_buffer_file.close()

        for neuron in self.neurons:
            neuron.post_run()
            if self.debug and not neuron.update_I_override:
                neuron._BaseNeuron__post_run()
                
        for synapse in self.synapses:
            synapse.post_run()



    def run_step(self, in_gpot_dict, in_spike_dict, out_gpot, out_spike):
        super(LPU, self).run_step(in_gpot_dict, in_spike_dict, \
                                  out_gpot, out_spike)

        # Try to read LPU input when connected to other LPUs and some data has
        # been received:
        if not self.run_on_myself:
            if (len(in_gpot_dict) +len(in_spike_dict)) != 0:
                self._read_LPU_input(in_gpot_dict, in_spike_dict)

        if self.update_other_rest:
            self.buffer.update_other_rest(in_gpot_dict, \
                np.sum(self.num_gpot_neurons), self.num_virtual_gpot_neurons)
            self.update_other_rest = False
        
        if self.update_flag:
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

            self.buffer.step()

        # Extract data to transmit to other LPUs:
        if not self.run_on_myself:
            self._extract_output(out_gpot, out_spike)
            #self.logger.info('out_spike: '+str(out_spike))
        # Save output data to disk:
        if self.update_flag and self.output:
            self._write_output()

        if not self.update_flag:
            self.update_flag = True
            if np.sum(self.num_input_gpot_neurons)>0:
                self.update_other_rest = True

    def _init_objects(self):
        self.neurons = [ self._instantiate_neuron(i,t,n) for i,(t,n) in enumerate(self.n_list) ]
        self.synapses = [ self._instantiate_synapse(i,t,n) for i,(t,n) in enumerate(self.s_list) ]

    def _setup_connectivity(self):

        def parse_interLPU_syn( pre_neu, pre_model, post_model ):
            """
            Insert parameters for synapses between neurons in this LPU and other LPUs.
            """

            #import ipdb; ipdb.set_trace()
            virtual_id = self.virtual_gpot_idx[-1] if pre_model=='gpot' else self.virtual_spike_idx[-1]
            public_id = self.public_gpot_list if post_model=='gpot' else self.public_spike_list
            for j, pre in enumerate( pre_neu ):
                pre_id = int(pre)
                post_neu = c.dest_idx(other_lpu, self.id, pre_model, post_model, src_ports=pre_id)
                for post in post_neu:
                    post_id = int(post)
                    num_syn = c.multapses( other_lpu,  pre_model,  pre_id,
                                            self.id, post_model, post_id)
                    for conn in range(num_syn):

                        # Get names of parameters associated with connection
                        # model:
                        s_model = c.get( other_lpu, pre_model, pre_id,
                                        self.id, post_model, post_id,
                                        conn=conn, param='model')
                        if s_model not in self.s_dict:
                            s = { k:[] for k in c.type_params[s_model] }
                            self.s_dict.update( {s_model:s} )
                        if not self.s_dict[s_model].has_key('pre'):
                            self.s_dict[s_model]['pre'] = []
                        if not self.s_dict[s_model].has_key('post'):
                            self.s_dict[s_model]['post'] = []
                        self.s_dict[s_model]['pre'].append( virtual_id[j] )
                        self.s_dict[s_model]['post'].append( public_id[post_id] )
                        for k,v in self.s_dict[s_model].items():
                            if k!='pre' and k!='post':
                                v.append( c.get(other_lpu, pre_model,  pre_id,
                                                self.id, post_model, post_id,
                                                conn=conn, param=k) )
        gpot_delay_steps = 0
        spike_delay_steps = 0


        order = self.order
        spike_shift = self.spike_shift

        self.other_lpu_map = {}
            
        if len(self._conn_dict)==0:
            self.update_flag = True
            self.update_other_rest = False
            self.run_on_myself = True
            self.num_virtual_gpot_neurons = 0
            self.num_virtual_spike_neurons = 0
            self.num_input_gpot_neurons = []
            self.num_input_spike_neurons = []

        else:
            self.run_on_myself = False
            
            # To use first execution tick to synchronize gpot resting potentials
            self.update_flag = False
            self._steps += 1
            self.update_other_rest = False

            self.num_input_gpot_neurons = []
            self.num_input_spike_neurons = []
            self.virtual_gpot_idx = []
            self.virtual_spike_idx = []
            self.input_spike_idx_map = []

            tmp1 = np.sum(self.num_gpot_neurons)
            tmp2 = np.sum(self.num_spike_neurons)


            for i,c in enumerate(self._conn_dict.itervalues()):
                other_lpu = c.B_id if self.id == c.A_id else c.A_id
                self.other_lpu_map[other_lpu] = i

                # parse synapse with gpot pre-synaptic neuron
                pre_gpot = c.src_idx(other_lpu, self.id, src_type='gpot')
                self.num_input_gpot_neurons.append(len(pre_gpot))

                self.virtual_gpot_idx.append(np.arange(tmp1,tmp1+\
                     self.num_input_gpot_neurons[-1]).astype(np.int32))
                tmp1 += self.num_input_gpot_neurons[-1]

                parse_interLPU_syn( pre_gpot, 'gpot', 'gpot' )
                parse_interLPU_syn( pre_gpot, 'gpot', 'spike')

                # parse synapse with spike pre-synaptic neuron
                self.input_spike_idx_map.append({})
                pre_spike = c.src_idx(other_lpu, self.id, src_type='spike')
                self.num_input_spike_neurons.append(len(pre_spike))

                self.virtual_spike_idx.append(np.arange(tmp2,tmp2+\
                     self.num_input_spike_neurons[-1]).astype(np.int32))
                tmp2 += self.num_input_spike_neurons[-1]

                for j,pre in enumerate(pre_spike):
                    self.input_spike_idx_map[i][int(pre)] = j

                parse_interLPU_syn( pre_spike, 'spike', 'gpot' )
                parse_interLPU_syn( pre_spike, 'spike', 'spike' )

            # total number of input graded potential neurons and spiking neurons
            self.num_virtual_gpot_neurons = int(np.sum(self.num_input_gpot_neurons))
            self.num_virtual_spike_neurons = int(np.sum(self.num_input_spike_neurons))

            # cumulative sum of number of input neurons
            # the purpose is to indicate position in the buffer
            self.cum_virtual_gpot_neurons = np.concatenate(((0,), \
                np.cumsum(self.num_input_gpot_neurons))).astype(np.int32)
            self.cum_virtual_spike_neurons = np.concatenate(((0,), \
                np.cumsum(self.num_input_spike_neurons))).astype(np.int32)



        cond_pre = []
        cond_post = []
        I_pre = []
        I_post = []
        reverse = []

        count = 0

        #import ipdb; ipdb.set_trace()
        self.total_gpot_neurons = self.my_num_gpot_neurons + \
                                            self.num_virtual_gpot_neurons
        self.total_spike_neurons = self.my_num_spike_neurons + \
                                            self.num_virtual_spike_neurons

        self.s_list = self.s_dict.items()
        num_synapses = [ len(s['id']) for t,s in self.s_list ]
        for (t,s) in self.s_list:
            order = np.argsort(s['post']).astype(np.int32)
            for k,v in s.items():
                v = np.asarray(v)[order]
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
                               |(cond_post < (self.idx_start_spike[i+1]+self.spike_shift)) )
                n['cond_post'] = cond_post[idx] - self.idx_start_spike[i] - self.spike_shift
                n['cond_pre'] = cond_pre[idx]
                n['reverse'] = reverse[idx]
                idx = np.where( (I_post >= self.idx_start_spike[i]+self.spike_shift) \
                               |(I_post < self.idx_start_spike[i+1]+self.spike_shift) )
                n['I_post'] = I_post[idx] - self.idx_start_spike[i] - spike_shift
                n['I_pre'] = I_pre[idx]
            else:
                idx = np.where( (cond_post >= self.idx_start_gpot[i]) \
                               |(cond_post < self.idx_start_gpot[i+1]) )
                n['cond_post'] = cond_post[idx] - self.idx_start_gpot[i]
                n['cond_pre'] = cond_pre[idx]
                n['reverse'] = reverse[idx]
                idx =  np.where( (I_post >= self.idx_start_gpot[i]) \
                                |(I_post < self.idx_start_gpot[i+1]) )
                n['I_post'] = I_post[idx] - self.idx_start_gpot[i]
                n['I_pre'] = I_pre[idx]

            n['num_dendrites_cond'] = Counter(n['cond_post'])
            n['num_dendrites_I'] = Counter(n['I_post'])

        self.gpot_delay_steps = int(round(gpot_delay_steps*1e-3 / self.dt)) + 1
        self.spike_delay_steps = int(round(spike_delay_steps*1e-3 / self.dt)) + 1



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

        if len(self.public_gpot_list)>0:
            self.public_gpot_list_g = garray.to_gpu(self.public_gpot_list)
            self.projection_gpot = garray.zeros(len(self.public_gpot_list), np.double)
            self._extract_gpot = self._extract_projection_gpot_func()

        if len(self.public_spike_list)>0:
            self.public_spike_list_g = garray.to_gpu( \
                (self.public_spike_list-self.spike_shift).astype(np.int32))
            self.projection_spike = garray.zeros(len(self.public_spike_list), np.int32)
            self._extract_spike = self._extract_projection_spike_func()



    def _read_LPU_input(self, in_gpot_dict, in_spike_dict):
        """
        Put inputs from other LPUs to buffer.

        """

        for other_lpu, gpot_data in in_gpot_dict.iteritems():
            i = self.other_lpu_map[other_lpu]
            if self.num_input_gpot_neurons[i] > 0:
                cuda.memcpy_htod(int(int(self.buffer.gpot_buffer.gpudata) \
                    +(self.buffer.gpot_current * self.buffer.gpot_buffer.ld \
                    + self.my_num_gpot_neurons + self.cum_virtual_gpot_neurons[i]) \
                    * self.buffer.gpot_buffer.dtype.itemsize), gpot_data)
                if self.debug:
                    self.in_gpot_files[other_lpu].root.array.append(gpot_data.reshape(1,-1))
            
        if self.debug:
            self.gpot_buffer_file.root.array.append(self.buffer.gpot_buffer.get().reshape(1,self.gpot_delay_steps,-1))

        #Will need to change this if only spike indexes are transmitted
        for other_lpu, sparse_spike in in_spike_dict.iteritems():
            i = self.other_lpu_map[other_lpu]
            if self.num_input_spike_neurons[i] > 0:
                full_spike = np.zeros(self.num_input_spike_neurons[i],dtype=np.int32)
                if len(sparse_spike)>0:
                    idx = np.asarray([self.input_spike_idx_map[i][k] \
                                      for k in sparse_spike], dtype=np.int32)
                    full_spike[idx] = 1

                cuda.memcpy_htod(int(int(self.buffer.spike_buffer.gpudata) \
                    +(self.buffer.spike_current * self.buffer.spike_buffer.ld \
                    + self.my_num_spike_neurons + self.cum_virtual_spike_neurons[i]) \
                    * self.buffer.spike_buffer.dtype.itemsize), full_spike)


    def _extract_output(self, out_gpot, out_spike, st=None):
        """
        This function should be changed so that if the output attribute is True,
        the following kernel calls are not made as all the GPU data will have to be
        transferred to the CPU anyways.
        """

        if self.num_public_gpot>0:
            self._extract_gpot.prepared_async_call(\
                self.grid_extract_gpot, self.block_extract, st, self.V.gpudata, \
                self.projection_gpot.gpudata, self.public_gpot_list_g.gpudata, \
                self.num_public_gpot)
        if self.num_public_spike>0:
            self._extract_spike.prepared_async_call(\
                self.grid_extract_spike, self.block_extract, st, self.spike_state.gpudata, \
                self.projection_spike.gpudata, self.public_spike_list_g.gpudata, \
                self.num_public_spike)

        # Save the states of the graded potential neurons and the indices of the
        # spiking neurons that have emitted a spike:
        if self.num_public_gpot>0:
            out_gpot.extend(self.projection_gpot.get())
        if self.num_public_spike>0:
            out_spike.extend(np.where(self.projection_spike.get())[0])

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
                h_ext = np.concatenate(h_ext, np.zeros(pad_shape), axis=0)
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

def neuron_cmp( x, y):
    if int(x[0]) < int(y[0]): return -1
    elif int(x[0]) > int(y[0]): return 1
    else: return 0

def synapse_cmp( x, y):
    if int(x[1]) < int(y[1]):
        return -1
    elif int(x[1]) > int(y[1]):
        return 1
    else:
        return 0

@property
def one_time_import(self):
    return self._one_time_import

@one_time_import.setter
def one_time_import(self, value):
    self._one_time_import = value

class circular_array:
    '''
    This class implements a circular buffer to support synapses with delays.
    Please refer the documentation of the template synapse class on information
    on how to access data correctly from this buffer
    '''
    def __init__(self, num_gpot_neurons, my_num_gpot_neurons, gpot_delay_steps,
                 rest, num_spike_neurons, spike_delay_steps):

        self.num_gpot_neurons = num_gpot_neurons
        if num_gpot_neurons > 0:
            self.dtype = np.double
            self.gpot_delay_steps = gpot_delay_steps
            self.gpot_buffer = parray.empty((gpot_delay_steps, num_gpot_neurons),np.double)

            self.gpot_current = 0
            
            if my_num_gpot_neurons>0:
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
