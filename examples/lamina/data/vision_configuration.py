#!/usr/bin/env python

"""
Lamina specification
"""

import csv
import os
import collections

import numpy as np
import gzip
import networkx as nx

class hex_array(object):
    """
       0  1  2  3   4
     ----------------------> cols (X=cols*sqrt(3))
    0| 0     2      4
     |    1     3
    1| 5     7      9
     |    6     8
    2| 10    12     14
     |    11    13
     |
     V
    rows (first col: 0,2,4,6)
    (Y=2*row if col is even else Y=2*row+1 )
    """
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.num_elements = nrows * ncols

        self.X = np.tile(np.arange(self.ncols, dtype = np.double).reshape((1, self.ncols))*np.sqrt(3),
                         (self.nrows, 1))
        if (self.ncols % 2 == 0):
            self.Y = np.tile(np.arange(2*self.nrows, dtype = np.double).reshape((self.nrows, 2)),
                             (1, self.ncols//2))
        else:
            self.Y = np.tile(np.arange(2*self.nrows, dtype = np.double).reshape((self.nrows, 2)),
                             (1, self.ncols//2+1))
            self.Y = self.Y[:,0:-1]
        self.col = np.tile(np.arange(self.ncols, dtype = np.int32).reshape((1, self.ncols)),
                           (self.nrows, 1))
        self.row = np.tile(np.arange(self.nrows, dtype = np.int32).reshape((self.nrows, 1)),
                           (1, self.ncols))


        self.col = self.col.reshape(-1)
        self.row = self.row.reshape(-1)
        self.num = np.arange(self.num_elements, dtype = np.int32).reshape(nrows, ncols)

    def find_neighbor(self, row, col):
        """
        neighbors are defined relatively as
            1
         2     6
            0
         3     5
            4
        """
        if col < 0 or col >= self.ncols:
            raise ValueError("column number " + str(col) + " exceeds array limit")
        if row < 0 or row >= self.nrows:
            raise ValueError("row number " + str(row) + " exceeds array limit")
        # adding neighbor 0 (self)
        neighbor = [self.num[row, col]]
        # adding neighbor 1
        neighbor.append(self.num[row-1, col] if row != 0 else None)
        # adding neighbor 2, 3
        if col == 0:
            neighbor.extend([None, None])
        elif col % 2 == 0:
            if row == 0:
                neighbor.extend([None, self.num[row, col-1]])
            else:
                neighbor.extend(list(self.num[row-1:row+1, col-1]))
        else:
            if row == self.nrows-1:
                neighbor.extend([self.num[row, col-1], None])
            else:
                neighbor.extend(list(self.num[row:row+2, col-1]))
        # adding neighbor 4
        neighbor.append(self.num[row+1, col] if row != self.nrows-1 else None)
        # adding neighbor 5, 6
        if col == self.ncols-1:
            neighbor.extend([None, None])
        elif col % 2 == 0:
            if row == 0:
                neighbor.extend([self.num[row, col+1], None])
            else:
                neighbor.extend(
                    list(self.num[row:row-2 if row-2 >= 0 else None:-1, col+1]))
        else:
            if row == self.nrows-1:
                neighbor.extend([None, self.num[row, col+1]])
            else:
                neighbor.extend(
                    list(self.num[row+1:row-1 if row-1 >= 0 else None:-1, col+1]))

        return neighbor


class vision_LPU(object):
    def __init__(self, nrows, ncols, neuron_csv,
                 columnar_synapse_csv, other_synapse_csv,
                 LPU_name):
        self.nrows = nrows
        self.ncols = ncols
        self.num_cartridges = nrows * ncols
        self.neuron_csv = neuron_csv
        self.columnar_synapse_csv = columnar_synapse_csv
        self.other_synapse_csv = other_synapse_csv
        self.hexarray = hex_array(nrows, ncols)
        self._connected = False
        self.LPU_name = LPU_name
        
        self.composition_rules = []

        # read in csv file and turn it into a numpy structured array
        neuron_list = []
        dtypes = [np.dtype('S10'), np.dtype('S32'),
                  np.dtype(np.int32), np.dtype(np.int32),
                  np.dtype(np.int32), np.dtype(np.int32),
                  np.dtype(np.int32), np.dtype(np.int32),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double)]
    
        with open(self.neuron_csv, 'rU') as csvfile:
            reader = csv.reader(csvfile)
            self.neuron_field_name = reader.next()
            n_entry = len(self.neuron_field_name)
            for row in reader:
                tmp = [dtypes[i].type(row[i]) for i in range(n_entry)]
                neuron_list.append(tuple(tmp))
        
        self.num_neuron_types = len(neuron_list)
        self.neuron_dict = np.array(
            neuron_list,
            dtype = [(a, b) for a, b in zip(self.neuron_field_name, dtypes)])
        
        # read in csv file and turn it into a numpy structured array
        if self.columnar_synapse_csv is not None:
            synapse_list = []
            dtypes = [np.dtype('S10'), np.dtype('S10'),
                      np.dtype('S32'),
                      np.dtype(np.int32), np.dtype(np.double),
                      np.dtype(np.double), np.dtype(np.double),
                      np.dtype(np.double), np.dtype(np.double),
                      np.dtype(np.double), np.dtype(np.double),
                      np.dtype(np.int32)]
            with open(self.columnar_synapse_csv, 'rU') as csvfile:
                reader = csv.reader(csvfile)
                self.synapse_field_name = reader.next()
                n_entry = len(self.synapse_field_name)
                for row in reader:
                    tmp = [dtypes[i].type(row[i]) for i in range(n_entry)]
                    synapse_list.append(tuple(tmp))
            
            self.num_synapse_types = len(synapse_list)
            self.synapse_dict = np.array(
                synapse_list,
                dtype = [(a, b) for a, b in zip(self.synapse_field_name, dtypes)])
        else:
            self.num_synapse_types = 0
            self.synapse_dict = []
        
        if self.other_synapse_csv is not None:
            synapse_list = []
            dtypes = [np.dtype('S10'), np.dtype('S10'),
                      np.dtype(np.int32),
                      np.dtype(np.int32), np.dtype(np.double),
                      np.dtype(np.double), np.dtype(np.double),
                      np.dtype(np.double), np.dtype(np.double),
                      np.dtype(np.double), np.dtype(np.double),
                      np.dtype(np.int32)]
            with open(self.columnar_synapse_csv, 'rU') as csvfile:
                reader = csv.reader(csvfile)
                self.synapse_field_name = reader.next()
                n_entry = len(self.synapse_field_name)
                for row in reader:
                    tmp = [dtypes[i].type(row[i]) for i in range(n_entry)]
                    synapse_list.append(tuple(tmp))
            
            self.num_synapse_types = len(synapse_list)
            self.synapse_dict = np.array(
                synapse_list,
                dtype = [(a, b) for a, b in zip(self.synapse_field_name, dtypes)])
        else:
            self.num_other_synapse_types = 0
            self.other_synapse_dict = []
        
    def create_cartridges(self):
        # create a number of cartridges
        self.cartridge_neuron_dict = self.neuron_dict[self.neuron_dict['columnar'] == 1]
        self.cartridge_synapse_dict = self.synapse_dict[self.synapse_dict['cart'] == 0]

        self.cartridges = []
        for _ in range(self.num_cartridges):
            self.cartridges.append(
                Cartridge(self.cartridge_neuron_dict,
                          self.cartridge_synapse_dict))
    
    def connect_cartridges(self):
        # connect cartridge from their neighbors
        if not hasattr(self, 'cartridges'):
            raise AttributeError("Need to create cartridges before connecting them")
        count = 0
        for cartridge in self.cartridges:
            row = np.asscalar(self.hexarray.row[count])
            col = np.asscalar(self.hexarray.col[count])
            cartridge.assign_pos(count, row, col,
                                 np.asscalar(self.hexarray.X[row,col]),
                                 np.asscalar(self.hexarray.Y[row,col]))
            neighbor_num = self.hexarray.find_neighbor(row, col)
            cartridge.set_neighbors(
                [self.cartridges[num] if num is not None else None
                for num in neighbor_num])
            count += 1
        self._connected = True

    def create_non_columnar_neurons(self):
        self.non_columnar_neurons = collections.OrderedDict()
        self.non_columnar_neuron_list = self.neuron_dict[self.neuron_dict['columnar'] != 1]
        
        dtnames = self.non_columnar_neuron_list.dtype.names
        for neuron_dict in self.non_columnar_neuron_list:
            name = neuron_dict['name']
            self.non_columnar_neurons.update({name: []})
            for _ in range(neuron_dict['columnar']):
                self.non_columnar_neurons[name].append(
                    Neuron(dict(zip(dtnames, [np.asscalar(p) for p in neuron_dict]))))

    def remove_cartridge(self, num):
        pass
    
    def remove_neuron_type(self, name):
        pass

    def __repr__(self):
        if hasattr(self, 'cartridges'):
            return 'LPU with '+str(len(self.cartridges))+' cartridges'
        else:
            return 'LPU unconfigured'

    def export_to_gexf(self, filename):
        g = nx.MultiDiGraph()
        num = 0
        
        for neuron_type in self.neuron_dict:
            if not neuron_type['dummy']:
                if neuron_type['columnar'] == 1:
                    name = neuron_type['name']
                    for cartridge in self.cartridges:
                        neuron = cartridge.neurons[name]
                        neuron.add_num(num)
                        neuron.process_before_export()
                        g.add_node(num, neuron.params)
                        num += 1

        for name in self.non_columnar_neurons.iterkeys():
            for neuron in self.non_columnar_neurons[name]:
                neuron.add_num(num)
                neuron.process_before_export()
                g.add_node(num, neuron.params)
                num += 1
        
        for cartridge in self.cartridges:
            for synapse in cartridge.synapses:
                synapse.process_before_export()
                g.add_edge(synapse.pre_neuron.num, synapse.post_neuron.num,
                           attr_dict = synapse.params)
    
        for cr in self.composition_rules:
            for synapse in cr['synapses']:
                synapse.process_before_export()
                g.add_edge(synapse.pre_neuron.num, synapse.post_neuron.num,
                           attr_dict = synapse.params)

        if isinstance(filename, str):
            name, ext = os.path.splitext(filename)
            if name == '':
                raise ValueError("Please specify a valid filename")
        
            if ext == '.gz':
                with gzip.open(filename, 'w') as f:
                    nx.write_gexf(g, f, prettyprint=True)
            else:
                if ext != '.gexf':
                    name = filename + '.gexf'
                else:
                    name = filename
                nx.write_gexf(g, name, prettyprint=True)
        else:
            raise ValueError("Specify the filename in string")

    def add_selectors(self):
        for neuron_type in self.neuron_dict:
            if not neuron_type['dummy']:
                if neuron_type['columnar'] == 1:
                    if neuron_type['public'] == 1:
                        name = neuron_type['name']
                        for cartridge in self.cartridges:
                            neuron = cartridge.neurons[name]
                            neuron.add_selector(
                                '/'+self.LPU_name+'/cart{0}'.format(cartridge.num)
                                +'/'+name)
        for name in self.non_columnar_neurons.iterkeys():
            count = 0
            for neuron in self.non_columnar_neurons[name]:
                if neuron.is_public():
                    neuron.add_selector(
                        '/'+self.LPU_name+'/'+name+'[{0}]'.format(count))
                    count += 1


class Lamina(vision_LPU):
    def __init__(self, nrows, ncols, neuron_csv,
                 columnar_synapse_csv, other_synapse_csv):
        super(Lamina, self).__init__(nrows, ncols, neuron_csv,
                                     columnar_synapse_csv, other_synapse_csv,
                                     'lamina')
    
    def connect_composition_II(self):
        # create synapses defined in composition rule II.
        if not self._connected:
            raise AttributeError("Need to connect cartridges before setting interconnects")
        self.rule2synapses = self.synapse_dict[self.synapse_dict['cart'] != 0]
        
        synapse_list = []
        
        dtnames = self.rule2synapses.dtype.names
        for cartridge in self.cartridges:
            for synapse_array in self.rule2synapses:
                neighbor_num = synapse_array['cart']
                if cartridge.neighbors[neighbor_num] is not None:
                    synapse = Synapse(
                        dict(zip(dtnames, [np.asscalar(p) for p in synapse_array])))
                    synapse.link(
                        cartridge.neurons[synapse_array['prename']],
                        cartridge.neighbors[neighbor_num].neurons[synapse_array['postname']])
                    synapse_list.append(synapse)
        self.composition_rules.append({'synapses': synapse_list})

    def connect_composition_I(self):
        am_list = self.non_columnar_neurons['Am']
        synapse_list = []
        
        n_amacrine = len(am_list) # self.non_columnar_neuron_number['Am']
        am_xpos = np.random.random(n_amacrine)*self.hexarray.X[-1,-1]
        am_ypos = np.random.random(n_amacrine)*self.hexarray.Y[-1,-1]
        count = 0
        for neuron in am_list:
            neuron.assign_pos(np.asscalar(am_xpos[count]),
                              np.asscalar(am_ypos[count]))
            count += 1

        bound = 4.0
        alpha_profiles = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
        fill = np.zeros((n_amacrine, self.num_cartridges), np.int32);
        count = 0
        for cartridge in self.cartridges:
            xpos = cartridge.xpos
            ypos = cartridge.ypos
            
            #calculate distance and find amacrine cells within 
            #distance defined by bound
            dist = np.sqrt((xpos-am_xpos)**2 + (ypos-am_ypos)**2)
            suitable_am = np.nonzero(dist <= bound)[0]
            # if less than 4 neurons in the bound, get
            # the 4 closest amacrine cells 
            if suitable_am.size < 4:
                suitable_am = np.argsort(dist)[0:4]
            
            for name in alpha_profiles:
                assigned = False
                for am_num in np.random.permutation(suitable_am):
                    if fill[am_num, count] < 3:
                        fill[am_num, count] += 1
                        #a1-a6 do not have synapses outside a cartridge
                        synapses = cartridge.replace_dummy(name, am_list[am_num])
                        synapse_list.extend(synapses)
                        assigned = True
                        break
                if not assigned:
                    print name + ' in cartridge ' + str(cartridge.num) + ' not assigned' 
            count += 1

        self.fill = fill
        self.composition_rules.append( {'synapses': synapse_list} )
        
    def __repr__(self):
        if hasattr(self, 'cartridges'):
            return 'Lamina LPU with '+str(len(self.cartridges))+' cartridges'
        else:
            return 'Lamina LPU unconfigured'


class Cartridge(object):
    def __init__(self, neuron, connection):
        self.connected = False
        self.neuron_list = neuron.copy()
        self.synapse_list = connection.copy()
        self.neurons = collections.OrderedDict()
        
        dtnames = self.neuron_list.dtype.names
    
        for neuron_dict in self.neuron_list:
            self.neurons.update(
                {neuron_dict['name']:
                 Neuron(dict(zip(dtnames, [np.asscalar(p) for p in neuron_dict])))})

        dtnames = self.synapse_list.dtype.names
        self.synapses = []
        for synapse_dict in self.synapse_list:
            synapse = Synapse(
                dict(zip(dtnames, [np.asscalar(p) for p in synapse_dict])))
            synapse.link(self.neurons[synapse.prename],
                         self.neurons[synapse.postname])
            self.synapses.append(synapse)
        
    def set_neighbors(self, neighbor_cartridges):
        self.neighbors = []
        for i in range(7):
            self.neighbors.append(neighbor_cartridges[i])
        
    def assign_pos(self, num, row, col, xpos, ypos):
        self.num = num
        self.row = row
        self.col = col
        self.xpos = xpos
        self.ypos = ypos
        for neurons in self.neurons:
            self.neurons[neurons].assign_pos(xpos, ypos)
        self.connected = True
    
    def position(self):
        return (self.xpos, self.ypos)

    def __repr__(self):
        if self.connected:
            return 'Cartridge at ' + str(self.position())
        else:
            return 'Isolated cartridge at '+ hex(id(self))
        
    def get_num(self):
        return self.num

    def get_xpos(self):
        return self.xpos

    def get_ypos(self):
        return self.ypos

    def replace_dummy(self, name, neuron):
        removed_synapse_list = []
        neuron_to_be_replaced = self.neurons[name]
        
        if neuron_to_be_replaced.params['dummy'] != 1:
            raise ValueError("Neuron to be replaced is not dummy element")
        
        for synapse in neuron_to_be_replaced.outgoing_synapses:
            flag = self.remove_synapse(synapse)
            synapse.replace_pre(neuron)
            if flag:
                removed_synapse_list.append(synapse)
        
        for synapse in neuron_to_be_replaced.incoming_synapses:
            flag = self.remove_synapse(synapse)
            synapse.replace_post(neuron)
            if flag:
                removed_synapse_list.append(synapse)
        
        self.neurons[name].set_parent(neuron)
        return removed_synapse_list
        
    def remove_neuron(self, name):
        self.neurons.pop(name)

    def remove_synapse(self, synapse):
        # the try/except here is to deal with Am to Am connection that
        # may have been removed previously by another Am in the same cartridge
        try:
            self.synapses.remove(synapse)
            return True
        except:
            return False

class Neuron(object):
    def __init__(self, param_dict):
        self.params = param_dict.copy()
        if self.params['model'] == 'MorrisLecar':
            spiking = False
        else:
            spiking = True
        self.params.update({'spiking': spiking})
        self.outgoing_synapses = []
        self.incoming_synapses = []
    
    @property
    def name(self):
        return self.params['name']
        
    def add_outgoing_synapse(self, synapse):
        self.outgoing_synapses.append(synapse)

    def add_incoming_synapse(self, synapse):
        self.incoming_synapses.append(synapse)
    
    def remove_outgoing_synapse(self, synapse):
        self.outgoing_synapses.remove(synapse)
    
    def remove_incoming_synapse(self, synapse):
        self.incoming_synapses.remove(synapse)

    def __repr__(self):
        return 'neuron '+self.params['name']+': '+str(self.params)

    def __str__(self):
        return 'neuron '+str(self.params['name'])

    def assign_pos(self, xpos, ypos):
        self.params.update({'xpos': xpos, 'ypos': ypos})
        
    def position(self):
            return (self.params['xpos'], self.params['ypos'])
        
    def add_num(self, num):
        self.num = num
        
    def process_before_export(self):
        self.params.update({'n_dendrites': len(self.incoming_synapses),
                            'n_outputs': len(self.outgoing_synapses)})
        if 'dummy' in self.params.keys():
            del self.params['dummy']
        if 'columnar' in self.params.keys():
            del self.params['columnar']
        self.params['input'] = bool(self.params['input'])
        self.params['output'] = bool(self.params['output'])
        self.params['public'] = bool(self.params['public'])
        self.params['extern'] = bool(self.params['extern'])
        self.params['model'] = str(self.params['model'])

    def is_dummy(self):
        if self.params.has_key('dummy'):
            return self.params['dummy']
        else:
            return False

    def is_public(self):
        return self.params['public']

    def add_selector(self, selector):
        self.params['selector'] = selector

    def set_parent(self, neuron):
        self.parent = neuron

class Synapse(object):
    def __init__(self, param_dict):
        self.params = param_dict.copy()
        self.params.update({'conductance': True})
    
    def link(self, pre_neuron, post_neuron):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.pre_neuron.add_outgoing_synapse(self)
        self.post_neuron.add_incoming_synapse(self)
        self.update_class(self.get_class(self.pre_neuron, self.post_neuron))

    def replace_pre(self, pre_neuron):
        self.pre_neuron = pre_neuron
        self.pre_neuron.add_outgoing_synapse(self)
        self.params['prename'] = pre_neuron.name

    def replace_post(self, post_neuron):
        self.post_neuron = post_neuron
        self.post_neuron.add_incoming_synapse(self)
        self.params['postname'] = post_neuron.name

    def __repr__(self):
        return ('synapse from '+self.params['prename']+' to ' + self.params['postname']
                + ': '+str(self.params))

    def __str__(self):
        return 'synapse '+str(self.params['prename'])+' to '+self.params['postname']
        
    def process_before_export(self):
        if 'cart' in self.params.keys():
            del self.params['cart']
        if 'scale' in self.params.keys():
            self.params['slope'] *= self.params['scale']
            self.params['saturation'] *= self.params['scale']
            del self.params['scale']
        self.params['model'] = str(self.params['model'])

    @staticmethod
    def get_class(preneuron, postneuron):
        """ preneuron: Neuron instance 
            postneuron: Neuron instance 
        """
        is_pre_spk = preneuron.params['spiking']
        is_post_spk = postneuron.params['spiking']
        
        if is_pre_spk and is_post_spk:
            return 0
        elif is_pre_spk and not is_post_spk:
            return 1
        elif not is_pre_spk and is_post_spk:
            return 2
        elif not is_pre_spk and not is_post_spk:
            return 3

    def update_class(self, cls):
        self.params.update({'class': cls})

    @property
    def prename(self):
        return self.params['prename']

    @property
    def postname(self):
        return self.params['postname']


def append_field(rec, name, arr, dtype=None):
    arr = np.asarray(arr)
    if dtype is None:
        dtype = arr.dtype
    newdtype = np.dtype(rec.dtype.descr + [(name, dtype)])
    newrec = np.empty(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    newrec[name] = arr
    return newrec

