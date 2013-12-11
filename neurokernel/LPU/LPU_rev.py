#!/usr/bin/env python
import collections

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

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

class LPU_rev(Module):

    @staticmethod
    def lpu_parser(filename):
        """
        GEXF-to-python LPU parser.

        Convert a .gexf LPU specifications into NetworkX graph type, and
        then pack data into list of dictionaries to be passed to the LPU
        module. The individual parameters will be represented by lists.

        Parameters
        ----------
        filename : String
            Filename containing LPU specification of GEXF format.
            See Notes for requirements to be met by the GEXF file.

        Returns
        -------
        n_dict : dict of dict of neuron
            The outer dict maps each neuron type to an inner dict which maps
            each attibutes of such neuron, ex. V1, to a list of data.

            ex. {'LeakyIAF':{'Vr':[...],'Vt':[...]},
                'MorrisLecar':{'V1':[...],'V2':[...]}}

        s_dict : dict of dict of synapse
            The outer dict maps each synapse type to an inner dict which maps
            each attribute of such synapse type to a list of data.

        Notes
        -----

        1. Each node(neuron) in the graph should necessarily have a boolean
           attribute called 'spiking' indicating whether the neuron is spiking
           or graded potential.
        2. Each node should have an integer attribute called 'type' indicating
           the model to be used for that neuron( Eg:- IAF, Morris-Lecar).
           Refer the documentation of LPU.neurons.BaseNeuron to implement
           custom neuron models.
        3. The attributes of the nodes should be consistent across all nodes
           of the same type. For example if a particular node of type 'IAF'
           has attribute 'bias', all nodes of type 'IAF' must necessarily
           have this attribute.
        4. Each node should have an boolean attribute called 'public',
           indicating whether that neuron either recieves input or provides
           output to other LPUs.
        5. Each node should have an boolean attribute called 'input' indicating
           whether the neuron accepts external input from a file.
        6. Each edge(synapse) in the graph should have an integer attribute
           called 'class' which should be one of the following values.
            0. spike-spike synapse
            1. spike-gpot synapse
            2. gpot-spike synapse
            3. gpot-gpot synapse
        7. Each edge should have an integer attribute called 'type' indicating
           the model to be used for that synapse( Eg:- alpha). Refer the
           documentation of LPU.synapses.BaseSynapse to implement custom
           synapse models.
        8. Each edge should have a boolean attribute called 'conductance'
           representing whether it's output is a conductance or current.
        10.For all edges with the 'conductance' attribute true, there should
           be an attribute called 'reverse'

        TODO
        ----
        Need to add code to assert all conditions mentioned above are met

        """

        # parse the GEXF file using networkX
        graph = nx.read_gexf(filename)

        # parse neuron data
        n_dict = {}
        neurons = graph.node.items()
        neurons.sort( cmp=neuron_cmp  )
        for id, neu in neurons:
            type = neu['type']
            # if the neuron type does not appear before, add the type into n_dict
            if type not in n_dict:
                n_dict[type] = { k:[] for k in neu.keys()+['id'] }
            # add neuron data into the subdictionary of n_dict
            for key in neu.iterkeys():
                n_dict[type][key].append( neu[key] )
            n_dict[type]['id'].append( id ) # id is a string in GXEF
        # remove duplicate type information
        for val in n_dict.itervalues(): val.pop('type')
        if not n_dict: n_dict = None

        # parse synapse data
        synapses = graph.edges(data=True)
        s_dict = {}
        synapses.sort(cmp=synapse_cmp)
        for syn in synapses:
            # syn[0/1]: pre-/post-neu id; syn[2]: dict of synaptic data
            type = syn[2]['type']
            syn[2]['id'] = int( syn[2]['id'] )
            # if the sysnapse type does not apear before, add it into s_dict
            if type not in s_dict:
                s_dict[type] = { k:[] for k in syn[2].keys()+['pre','post'] }
            # add sysnaptic data into the subdictionary of s_dict
            for key in syn[2].iterkeys():
                s_dict[type][key].append( syn[2][key] )
            s_dict[type]['pre'].append( syn[0] )
            s_dict[type]['post'].append( syn[1] )
        for val in s_dict.itervalues(): val.pop('type')
        if not s_dict: s_dict = {}
        return n_dict, s_dict

    def __init__(self, dt, n_dict, s_dict, input_file=None, device=0,
                 output_file=None, port_ctrl=base.PORT_CTRL,
                 port_data=base.PORT_DATA, id=None, debug=False):

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
        id : string
            Name of the LPU
        debug : boolean
            This parameter will be passed to all the neuron and synapse
            objects instantiated by this LPU and is intended to be used
            for debugging purposes only.
            Will be set to False by default

        """

        super(LPU_rev, self).__init__(port_data=port_data, port_ctrl=port_ctrl,
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

        #TODO: comment
        self.n_list = n_dict.items()
        n_type_is_spk = [ n['spiking'][0] for t,n in self.n_list ]
        n_type_num = [ len(n['id']) for t,n in self.n_list ]
        n_id = np.array(sum( [ n['id'] for t,n in self.n_list ], []), dtype=np.int32)
        n_is_spk = np.array(sum( [ n['spiking'] for t,n in self.n_list ], []))
        n_is_pub = np.array(sum( [ n['public'] for t,n in self.n_list ], []))
        n_has_in = np.array(sum( [ n['input'] for t,n in self.n_list ], []))

        #TODO: comment
        self.num_gpot_neurons = np.where( n_type_is_spk, 0, n_type_num)
        self.num_spike_neurons = np.where( n_type_is_spk, n_type_num, 0)
        self.my_num_gpot_neurons = sum( self.num_gpot_neurons )
        self.my_num_spike_neurons = sum( self.num_spike_neurons )
        self.gpot_idx = n_id[ ~n_is_spk ]
        self.spike_idx = n_id[ n_is_spk ]
        self.order = np.argsort( np.concatenate( (self.gpot_idx, self.spike_idx )) )
        self.gpot_order = np.argsort( self.gpot_idx )
        self.spike_order = np.argsort( self.spike_idx )
        self.spike_shift = self.my_num_gpot_neurons
        self.input_neuron_list = self.order[ n_id[ n_has_in ] ]
        self.public_spike_list = self.order[ n_id[ n_is_pub & n_is_spk ] ]
        self.public_gpot_list = self.order[ n_id[ n_is_pub & ~n_is_spk ] ]
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
        super(LPU_rev,self).pre_run()
        self._setup_connectivity()
        self._initialize_gpu_ds()
        self._init_objects()
        self.buffer = circular_array(self.total_gpot_neurons, \
                        self.gpot_delay_steps, self.V, \
                        self.total_spike_neurons, self.spike_delay_steps)

        if self.input_file:
            self.input_h5file = tables.openFile(self.input_file)

            self.one_time_import = 10
            self.file_pointer = 0
            self.I_ext = parray.to_gpu(self.input_h5file.root.array.read(\
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
        super(LPU_rev,self).post_run()
        if self.output:
            if self.my_num_gpot_neurons > 0:
                self.output_gpot_file.close()
            if self.my_num_spike_neurons > 0:
                self.output_spike_file.close()

        for neuron in self.neurons:
            neuron.post_run()

        for synapse in self.synapses:
            synapse.post_run()



    def run_step(self, in_gpot_dict, in_spike_dict, out_gpot, out_spike):
        super(LPU_rev, self).run_step(in_gpot_dict, in_spike_dict, \
                                  out_gpot, out_spike)

        # Try to read LPU input when connected to other LPUs and some data has
        # been received:
        if not self.run_on_myself:
            if (len(in_gpot_dict) +len(in_spike_dict)) != 0:
                self._read_LPU_input(in_gpot_dict, in_spike_dict)

        if self.update_resting_potential_history and \
          (len(in_gpot_dict) +len(in_spike_dict)) != 0:
            self.buffer.update_other_rest(in_gpot_dict, \
                np.sum(self.num_gpot_neurons), self.num_virtual_gpot_neurons)
            self.update_resting_potential_history = False

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
        if self.output:
            self._write_output()



    def _init_objects(self):
        self.neurons = [ self._instantiate_neuron(i,t,n) for i,(t,n) in enumerate(self.n_list) ]
        self.synapses = [ self._instantiate_synapse(i,t,n) for i,(t,n) in enumerate(self.s_list) ]

    def _setup_connectivity(self):

        def parse_interLPU_syn( pre_neu, pre_type, post_type ):
            """
            Insert parameters for synapses between neurons in this LPU and other LPUs.
            """

            #import ipdb; ipdb.set_trace()
            virtual_id = self.virtual_gpot_idx[-1] if pre_type=='gpot' else self.virtual_spike_idx[-1]
            public_id = self.public_gpot_list if post_type=='gpot' else self.public_spike_list
            for j, pre in enumerate( pre_neu ):
                pre_id = int(pre)
                post_neu = c.dest_idx(other_lpu, self.id, pre_type, post_type, src_ports=pre_id)
                for post in post_neu:
                    post_id = int(post)
                    num_syn = c.multapses( other_lpu,  pre_type,  pre_id,
                                            self.id, post_type, post_id)
                    for conn in range(num_syn):

                        # Get names of parameters associated with connection type:
                        s_type = c.get( other_lpu, pre_type, pre_id,
                                        self.id, post_type, post_id,
                                        conn=conn, param='type')
                        if s_type not in self.s_dict:
#                            s = { k:[] for k in c.get_dict_params(s_type).keys() }
#                            self.s_dict.update( {s_type:s} )
                            s = { k:[] for k in c.type_params[s_type] }
                            self.s_dict.update( {s_type:s} )
                        if not self.s_dict[s_type].has_key('pre'):
                            self.s_dict[s_type]['pre'] = []
                        if not self.s_dict[s_type].has_key('post'):
                            self.s_dict[s_type]['post'] = []
                        self.s_dict[s_type]['pre'].append( virtual_id[j] )
                        self.s_dict[s_type]['post'].append( public_id[post_id] )
                        for k,v in self.s_dict[s_type].items():
                            if k!='pre' and k!='post':
                                v.append( c.get(other_lpu, pre_type,  pre_id,
                                                self.id, post_type, post_id,
                                                conn=conn, param=k) )
        gpot_delay_steps = 0
        spike_delay_steps = 0


        order = self.order
        spike_shift = self.spike_shift

        if len(self._conn_dict)==0:
            self.update_resting_potential_history = False
            self.run_on_myself = True
            self.num_virtual_gpot_neurons = 0
            self.num_virtual_spike_neurons = 0
            self.num_input_gpot_neurons = 0
            self.num_input_spike_neurons = 0

        else:
            self.update_resting_potential_history = True
            self.run_on_myself = False

            self.num_input_gpot_neurons = []
            self.num_input_spike_neurons = []
            self.virtual_gpot_idx = []
            self.virtual_spike_idx = []

            tmp1 = np.sum(self.num_gpot_neurons)
            tmp2 = np.sum(self.num_spike_neurons)



            for i,c in enumerate(self._conn_dict.itervalues()):
                other_lpu = c.B_id if self.id == c.A_id else c.A_id
                # parse synapse with gpot pre-synaptic neuron
                pre_gpot = c.src_idx(other_lpu, self.id, src_type='gpot')

                self.num_input_gpot_neurons.append(len(pre_gpot))

                self.virtual_gpot_idx.append(np.arange(tmp1,tmp1+\
                     self.num_input_gpot_neurons[-1]).astype(np.int32))
                tmp1 += self.num_input_gpot_neurons[-1]

                parse_interLPU_syn( pre_gpot, 'gpot', 'gpot' )
                parse_interLPU_syn( pre_gpot, 'gpot', 'spike')

                # parse synapse with spike pre-synaptic neuron
                pre_spike = c.src_idx(other_lpu, self.id, src_type='spike')
                self.num_input_spike_neurons.append(len(pre_spike))

                self.virtual_spike_idx.append(np.arange(tmp2,tmp2+\
                     self.num_input_spike_neurons[-1]).astype(np.int32))
                tmp2 += self.num_input_spike_neurons[-1]

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
            order = np.argsort(s['post'])
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

        self.total_synapses = np.sum(num_synapses)
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
            self.V = garray.zeros(int(1), np.float64)

        if self.my_num_spike_neurons>0:
            self.spike_state = garray.zeros(int(self.my_num_spike_neurons), np.int32)

        if len(self.public_gpot_list)>0:
            self.public_gpot_list_g = garray.to_gpu(self.public_gpot_list)
            self.projection_gpot = garray.zeros(len(self.public_gpot_list), np.double)
            self._extract_gpot = self._extract_projection_gpot_func()

        if len(self.public_spike_list)>0:
            self.public_spike_list_g = garray.to_gpu(self.public_spike_list-self.spike_shift)
            self.projection_spike = garray.zeros(len(self.public_spike_list), np.int32)
            self._extract_spike = self._extract_projection_spike_func()



    def _read_LPU_input(self, in_gpot_dict, in_spike_dict):
        """
        Put inputs from other LPUs to buffer.

        """

        for i, gpot_data in enumerate(in_gpot_dict.itervalues()):
            if self.num_input_gpot_neurons[i] > 0:
                cuda.memcpy_htod(int(int(self.buffer.gpot_buffer.gpudata) \
                    +(self.buffer.gpot_current * self.buffer.gpot_buffer.ld \
                    + self.my_num_gpot_neurons + self.cum_virtual_gpot_neurons[i]) \
                    * self.buffer.gpot_buffer.dtype.itemsize), gpot_data)


        #Will need to change this if only spike indexes are transmitted
        for i, spike_data in enumerate(in_spike_dict.itervalues()):
            if self.num_input_spike_neurons[i] > 0:
                cuda.memcpy_htod(int(int(self.buffer.spike_buffer.gpudata) \
                    +(self.buffer.spike_current * self.buffer.spike_buffer.ld \
                    + self.my_num_spike_neurons + self.cum_virtual_spike_neurons[i]) \
                    * self.buffer.spike_buffer.dtype.itemsize), spike_data)


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
        if self.input_eof:
            return
        cuda.memcpy_dtod(int(int(self.synapse_state.gpudata) + \
            self.total_synapses*self.synapse_state.dtype.itemsize), \
            int(int(self.I_ext.gpudata) + self.frame_count*self.I_ext.ld*self.I_ext.dtype.itemsize), \
            self.num_input * self.synapse_state.dtype.itemsize)
        self.frame_count += 1
        if self.frame_count >= self.one_time_import:
            h_ext = self.input_h5file.root.array.read(self.file_pointer, self.file_pointer + self.one_time_import)
            if h_ext.shape[0] == self.I_ext.shape[0]:
                self.I_ext.set(h_ext)
                self.file_pointer += self.one_time_import
                self.frame_count = 0
            else:
                if self.file_pointer == self.input_h5file.root.array.shape[0]:
                    self.logger.info('Input end of file reached. Behaviour is ' +\
                                    'undefined for subsequent steps')
                    self.input_eof = True

    #TODO
    def _update_buffer(self):
        if self.total_gpot_neurons>0:
            cuda.memcpy_dtod(int(self.buffer.gpot_buffer.gpudata) + \
                self.buffer.gpot_current*self.buffer.gpot_buffer.ld* \
                self.buffer.gpot_buffer.dtype.itemsize, self.V.gpudata, \
                self.V.nbytes)
        if self.total_spike_neurons>0:
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
            self.logger.info('Error instantiating neurons of type ' + t )
            return []

        if n['spiking'][0]:
            neuron = self._neuron_classes[ind]( n, int(int(self.spike_state.gpudata) + \
                        self.spike_state.dtype.itemsize*self.idx_start_spike[i]), \
                        self.dt, debug=self.debug)
        else:
            neuron = self._neuron_classes[ind](n, int(self.V.gpudata) +  \
                        self.V.dtype.itemsize*self.idx_start_gpot[i], \
                        self.dt, debug=self.debug)

        if not neuron.update_I_override:
            baseneuron.BaseNeuron.__init__(neuron, n, int(int(self.V.gpudata) +  \
                        self.V.dtype.itemsize*self.idx_start_gpot[i]), \
                        self.dt, debug=self.debug)

        return neuron

    #TODO
    def _instantiate_synapse(self, i, t, s):
        try:
            ind = self._synapse_names.index( t )
            # ind = int(t)
        except:
            self.logger.info('Error instantiating synapses of type ' + t )
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

    def update_other_rest(self, gpot_data, my_num_gpot_neurons, num_virtual_gpot_neurons):
        if self.num_gpot_neurons > 0:
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
