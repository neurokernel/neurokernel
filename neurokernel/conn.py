#!/usr/bin/env python

"""
Synaptic connectivity class.
"""

import collections
import re
import string

import numpy as np
import scipy.sparse
import scipy as sp

class Connectivity(object):
    """
    Inter-LPU connectivity.

    Stores the connectivity between two LPUs as a series of sparse matrices.
    Every connection in an instance of the class has the following indices:

    - source port ID
    - destination port ID
    - synapse number (when two neurons are connected by more than one neuron)
    - direction ('+' for source to destination, '-' for destination to source)
    - parameter name (the default is 'conn' for simple connectivity)

    Examples
    --------
    The first connection between port 0 in one LPU with port 3 in some other LPU can
    be accessed as c[0,3,0,'+'].

    Notes
    -----
    Since connections between LPUs should necessarily not contain any recurrent
    connections, it is more efficient to store the inter-LPU connections in two
    separate matrices that respectively map to and from the ports in each LPU
    rather than a large matrix whose dimensions comprise the total number of
    ports in both LPUs.
    
    """

    def __init__(self, n1, n2):

        # The number of ports in both of the LPUs must be nonzero:
        assert n1 != 0
        assert n2 != 0

        self.shape = (n1, n2)
        
        # All matrices are stored in this dict:
        self._data = {}

        # Keys corresponding to each connectivity direction are stored in the
        # following lists:
        self._keys_by_dir = {'+': [],
                             '-': []}

        # Create connectivity matrices for both directions:
        key = self._make_key(0, '+', 'conn')
        self._data[key] = self._make_matrix(self.shape, int)
        self._keys_by_dir['+'].append(key)        
        key = self._make_key(0, '-', 'conn')
        self._data[key] = self._make_matrix(self.shape, int)
        self._keys_by_dir['-'].append(key)

    @property
    def N_src(self):
        """
        Number of source ports.
        """
        
        return self.shape[0]

    @property
    def N_dest(self):
        """
        Number of destination ports.
        """
        
        return self.shape[1]

    @property
    def max_multapses(self):
        """
        Maximum number of multapses that can be stored per neuron pair.
        """

        result = 0
        for dir in ['+', '-']:
            count = 0
            for key in self._keys_by_dir[dir]:
                if re.match('.*\/%s\/conn' % re.escape(dir), key):
                    count += 1
            if count > result:
                result = count
        return result

    @property
    def src_connected_mask(self):
        """
        Mask of source neurons with connections to destination neurons.
        """
        
        m_list = [self._data[k] for k in self._keys_by_dir['+']]
        return np.any(np.sum(m_list).toarray(), axis=1)
                      
    @property
    def src_connected_idx(self):
        """
        Indices of source neurons with connections to destination neurons.
        """
        
        return np.arange(self.shape[1])[self.src_connected_mask]
    
    @property
    def nbytes(self):
        """
        Approximate number of bytes required by the class instance.

        Notes
        -----
        Only accounts for nonzero values in sparse matrices.
        """

        count = 0
        for key in self._data.keys():
            count += self._data[key].dtype.itemsize*self._data[key].nnz
        return count
    
    def _format_bin_array(self, a, indent=0):
        """
        Format a binary array for printing.
        
        Notes
        -----
        Assumes a 2D array containing binary values.
        """
        
        sp0 = ' '*indent
        sp1 = sp0+' '
        a_list = a.toarray().tolist()
        if a.shape[0] == 1:
            return sp0+str(a_list)
        else:
            return sp0+'['+str(a_list[0])+'\n'+''.join(map(lambda s: sp1+str(s)+'\n', a_list[1:-1]))+sp1+str(a_list[-1])+']'
        
    def __repr__(self):
        result = 'src -> dest\n'
        result += '-----------\n'
        for key in self._keys_by_dir['+']:
            result += key + '\n'
            result += self._format_bin_array(self._data[key]) + '\n'
        result += '\ndest -> src\n'
        result += '-----------\n'
        for key in self._keys_by_dir['-']:
            result += key + '\n'
            result += self._format_bin_array(self._data[key]) + '\n'
        return result
        
    def _make_key(self, syn, dir, param):
        """
        Create a unique key for a matrix of synapse properties.
        """
        
        return string.join(map(str, [syn, dir, param]), '/')

    def _make_matrix(self, shape, dtype=np.double):
        """
        Create a sparse matrix of the specified shape.
        """
        
        return sp.sparse.lil_matrix(shape, dtype=dtype)
            
    def get(self, source, dest, syn=0, dir='+', param='conn'):
        """
        Retrieve a value in the connectivity class instance.
        """

        assert type(syn) == int
        assert dir in ['-', '+']
        
        result = self._data[self._make_key(syn, dir, param)][source, dest]
        if not np.isscalar(result):
            return result.toarray()
        else:
            return result

    def set(self, source, dest, syn=0, dir='+', param='conn', val=1):
        """
        Set a value in the connectivity class instance.

        Notes
        -----
        Creates a new storage matrix when the one specified doesn't exist.        
        """

        assert type(syn) == int
        assert dir in ['-', '+']
        
        key = self._make_key(syn, dir, param)
        if not self._data.has_key(key):

            # XX should ensure that inserting a new matrix for an existing param
            # uses the same type as the existing matrices for that param XX
            self._data[key] = self._make_matrix(self.shape, type(val))
            self._keys_by_dir[dir].append(key)
        self._data[key][source, dest] = val

    def flip(self):
        """
        Returns an object instance with the source and destination LPUs flipped.
        """

        c = Connectivity(self.shape[::-1])
        for old_key in self._data.keys():

            # Reverse the direction in the key:
            key_split = old_key.split('/')
            old_dir = key_split[1]
            if old_dir == '+':
                new_dir = '-'
            elif old_dir == '-':
                new_dir = '+'
            else:
                raise ValueError('invalid direction in key')    
            key_split[1] = new_dir
            new_key = '/'.join(key_split)
            c._data[new_key] = self._data[old_key].T
            c._keys_by_dir[new_dir].append(new_key)
        return c
        
    def __getitem__(self, s):        
        return self.get(*s)

    def __setitem__(self, s, val):
        self.set(*s, val=val)
    
class IntervalIndex(object):
    """
    Converts between indices within intervals of a sequence and absolute indices.

    When an instance of this class is indexed by an integer without
    specification of any label, the index is assumed to be absolute and
    converted to a relative index. If a label is specified, the index is assumed
    to be relative and is converted to an absolute index.
    
    Example
    -------
    >>> idx = IntervalIndex([0, 5, 10], ['a', 'b'])
    >>> idx[3]
    3
    >>> idx[7]
    2
    >>> idx['b', 2]
    7
    >>> idx['a', 2:5]
    slice(2, 5, None)
    >>> idx['b', 2:5]
    slice(7, 10, None)
    
    Parameters
    ----------
    bounds : list of int
        Boundaries of intervals represented as a sequence. For example,
        [0, 5, 10] represents the intervals (0, 5) and (5, 10) in the sequence
        range(0, 10).
    labels : list
        Labels to associate with each of the intervals. len(labels) must be
        one less than len(bounds).

    Notes
    -----
    Conversion from absolute to relative indices is not efficient for sequences
    of many intervals.
    
    """
    
    def __init__(self, bounds, labels):
        if len(labels) != len(bounds)-1:
            raise ValueError('incorrect number of labels')
        self._intervals = collections.OrderedDict()
        self._bounds = collections.OrderedDict()
        self._full_interval = min(bounds), max(bounds)
        for i in xrange(len(bounds)-1):
            if bounds[i+1] <= bounds[i]:
                raise ValueError('bounds sequence must be monotonic increasing')
            self._intervals[labels[i]] = (0, bounds[i+1]-bounds[i])
            self._bounds[labels[i]] = bounds[i]

    def __repr__(self):
        len_bound_min = str(max(map(lambda interval, bound: len(str(interval[0]+bound)),
                                  self._intervals.values(),
                                  self._bounds.values())))
        len_bound_max = str(max(map(lambda interval, bound: len(str(interval[1]+bound)),
                                  self._intervals.values(),
                                  self._bounds.values())))
        len_label = str(max(map(lambda x: len(str(x)), self._intervals.keys())))
        result = ''
        for label in self._intervals.keys():
            interval = self._intervals[label]
            bound = self._bounds[label]
            result += ('%-'+len_label+'s: (%-'+len_bound_min+'i, %'+len_bound_max+'i)') % \
              (str(label), interval[0]+bound, interval[1]+bound)
            if label != self._intervals.keys()[-1]:
                result += '\n'
        return result
        
    def _validate(self, i, interval):
        """
        Validate an index or slice against a specified interval.
        """

        if type(i) == int:
            if i < interval[0] or i >= interval[1]:
                raise ValueError('invalid index')
        elif type(i) == slice:
            if i.start < interval[0] or i.stop > interval[1]:
                raise ValueError('invalid slice')
        else:
            raise ValueError('invalid type')
        
    def __getitem__(self, i):
                    
        # If a tuple is specified, the first entry is assumed to be the interval label:
        if type(i) == tuple:
            label, idx = i
            self._validate(idx, self._intervals[label])
            if type(idx) == int:
                return idx+self._bounds[label]
            else:
                return slice(idx.start+self._bounds[label],
                             idx.stop+self._bounds[label],
                             idx.step)
        elif type(i) == int:
            for label in self._intervals.keys():
                interval = self._intervals[label]
                bound = self._bounds[label]
                if i >= interval[0]+bound and i < interval[1]+bound:
                    return i-(interval[0]+bound)
        elif type(i) == slice:
            for label in self._intervals.keys():
                interval = self._intervals[label]
                bound = self._bounds[label]
                if i.start >= interval[0]+bound and i.stop <= interval[1]+bound:
                    return slice(i.start-(interval[0]+bound),
                                 i.stop-(interval[0]+bound),
                                 i.step)            
            raise NotImplementedError('unsupported conversion of absolute to '
                                      'relative slices')
        else:
            raise ValueError('unrecognized type')

class MixedConnectivity(Connectivity):
    """
    Inter-LPU connectivity with support for graded potential and spiking
    neurons.

    Parameters
    ----------
    src_gpot : int
        Number of source graded potential neurons.
    src_spike : int
        Number of source spiking neurons.
    dest_gpot : int
        Number of destination graded potential neurons.
    dest_spike : int
        Number of destination spiking neurons.
        
    """
    
    def __init__(self, src_gpot, src_spike, dest_gpot, dest_spike):
        self.n_gpot = [src_gpot, dest_gpot]
        self.n_spike = [src_spike, dest_spike]
        super(MixedConnectivity, self).__init__(src_gpot+src_spike,
                                                dest_gpot+dest_spike)

        # Create index translators to enable use of separate sets of identifiers
        # for graded potential and spiking neurons:
        self.idx_translate = []
        for i in xrange(2):
            if self.n_gpot[i] == 0:
                idx_translate = IntervalIndex([0, self.n_gpot[i]], ['spike'])
            elif self.n_spike[i] == 0:
                idx_translate = IntervalIndex([0, self.n_gpot[i]], ['gpot'])
            else:
                idx_translate = IntervalIndex([0, self.n_gpot[i], self.n_gpot[i]+self.n_spike[i]],
                                                ['gpot', 'spike'])
            self.idx_translate.append(idx_translate)

    def get(self, source_type, source, dest_type, dest,
            syn=0, dir='+', param='conn'):
        """
        Retrieve a value in the connectivity class instance.
        """

        assert source_type in ['gpot', 'spike']
        assert dest_type in ['gpot', 'spike']
        s = self.idx_translate[0][source_type, source], \
            self.idx_translate[1][dest_type, dest], \
            syn, dir, param    
        return super(MixedConnectivity, self).get(*s)

    def set(self, source_type, source, dest_type, dest,
            syn=0, dir='+', param='conn', val=1):
        assert source_type in ['gpot', 'spike']
        assert dest_type in ['gpot', 'spike']
        s = self.idx_translate[0][source_type, source], \
            self.idx_translate[1][dest_type, dest], \
            syn, dir, param    
        return super(MixedConnectivity, self).set(*s, val=val)
    
    def __repr__(self):
        return super(MixedConnectivity, self).__repr__()+\
          '\nsrc idx\n'+self.idx_translate[0].__repr__()+\
          '\n\ndest idx\n'+self.idx_translate[1].__repr__()
            
if __name__ == '__main__':
    pass
