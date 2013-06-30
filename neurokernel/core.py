#!/usr/bin/env python

"""
Core Neurokernel classes.
"""

import atexit
import collections
import numpy as np
import time

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import twiggy
import bidict

import base
from ctx_managers import IgnoreKeyboardInterrupt, OnKeyboardInterrupt, \
     ExceptionOnSignal, TryExceptionOnSignal
from tools.misc import rand_bin_matrix, catch_exception
        
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
    >>> idx['b', :]
    slice(5, 10, None)
        
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
            
            # Slices such as :, 0:, etc. are deemed valid:
            if (i.start < interval[0] and i.start is not None) or \
              (i.stop > interval[1] and i.stop is not None):
                raise ValueError('invalid slice')
        else:
            raise ValueError('invalid type')
        
    def __getitem__(self, i):
                    
        # If a tuple is specified, the first entry is assumed to be the interval
        # label:        
        if type(i) == tuple:
            label, idx = i
            self._validate(idx, self._intervals[label])
            if type(idx) == int:
                return idx+self._bounds[label]
            else:

                # Handle cases where one of the slice bounds is None:
                if idx.start is None:
                    start = self._bounds[label]
                else:
                    start = idx.start+self._bounds[label]
                if idx.stop is None:
                    stop = self._bounds[label]+self._intervals[label][1]
                else:
                    stop = idx.stop+self._bounds[label]
                return slice(start, stop, idx.step)                             
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

class Connectivity(base.BaseConnectivity):
    """
    Intermodule connectivity with support for graded potential and spiking
    neurons.

    Stores the connectivity between two LPUs as a series of sparse matrices.
    Every entry in an instance of the class has the following indices:

    - source module ID (must be defined upon class instantiation)
    - source neuron type ('gpot', 'spike', or 'all')
    - source neuron ID
    - destination module ID (must be defined upon class instantiation)
    - destination neuron type ('gpot', 'spike', or 'all')
    - destination neuron ID
    - connection number (when two ports are connected by more than one connection)
    - parameter name (the default is 'conn' for simple connectivity)
 
    Each connection may therefore have several parameters; parameters associated
    with nonexistent connections (i.e., those whose 'conn' parameter is set to
    0) should be ignored.
    
    Parameters
    ----------
    N_A_gpot : int
        Number of graded potential neurons to interface with on module A.
    N_A_spike : int
        Number of spiking neurons to interface with on module A.
    N_B_gpot : int
        Number of graded potential neurons to interface with on module B.
    N_B_spike : int
        Number of destination spiking neurons to interface with on module A.
    N_mult: int
        Maximum supported number of connections between any two neurons
        (default 1). Can be raised after instantiation.    
    A_id : str
        First module ID (default 'A').
    B_id : str
        Second module ID (default 'B').

    Examples
    --------
    The first connection between spiking neuron 0 in one LPU with graded
    potential 3 in some other LPU can be accessed as
    c['A','spike',0,'B','gpot',3,0]. The 'weight' parameter associated with this
    connection can be accessed as c['A','spike',0,'B','gpot',3,0,'weight']

    """
    
    def __init__(self, N_A_gpot, N_A_spike, N_B_gpot, N_B_spike,
                 N_mult=1, A_id='A', B_id='B'):
        self.N_A_gpot = N_A_gpot
        self.N_A_spike = N_A_spike
        self.N_B_gpot = N_B_gpot
        self.N_B_spike = N_B_spike
        super(Connectivity, self).__init__(N_A_gpot+N_A_spike,
                                           N_B_gpot+N_B_spike, N_mult,
                                           A_id, B_id)
            
        # Create index translators to enable use of separate sets of identifiers
        # for graded potential and spiking neurons:
        self.idx_translate = {}
        if self.N_A_gpot == 0:
            self.idx_translate[A_id] = \
                IntervalIndex([0, self.N_A_spike], ['spike'])
        elif self.N_A_spike == 0:
            self.idx_translate[A_id] = \
                IntervalIndex([0, self.N_A_gpot], ['gpot'])
        else:
            self.idx_translate[A_id] = \
                IntervalIndex([0, self.N_A_gpot,
                               self.N_A_gpot+self.N_A_spike],
                              ['gpot', 'spike'])
        if self.N_B_gpot == 0:
            self.idx_translate[B_id] = \
                IntervalIndex([0, self.N_B_spike], ['spike'])
        elif self.N_B_spike == 0:
            self.idx_translate[B_id] = \
                IntervalIndex([0, self.N_B_gpot], ['gpot'])
        else:
            self.idx_translate[B_id] = \
                IntervalIndex([0, self.N_B_gpot,
                               self.N_B_gpot+self.N_B_spike],
                              ['gpot', 'spike'])
            
    def N(self, id, n_type='all'):
        """
        Return number of neurons of the specified type associated with the specified module.
        """

        if n_type == 'all':
            return super(Connectivity, self).N(id)
        elif n_type == 'gpot':
            return self.N_gpot(id)
        elif n_type == 'spike':
            return self.N_spike(id)
        else:
            raise ValueError('invalid neuron type')

    def N_spike(self, id):
        """
        Return number of spiking neurons associated with the specified module.
        """

        if id == self.A_id:
            return self.N_A_spike
        elif id == self.B_id:
            return self.N_B_spike
        else:
            raise ValueError('invalid module ID')

    def N_gpot(self, id):
        """
        Return number of graded potential neurons associated with the specified module.
        """

        if id == self.A_id:
            return self.N_A_gpot
        elif id == self.B_id:
            return self.N_B_gpot
        else:
            raise ValueError('invalid module ID')
        
    def src_mask(self, src_id='', dest_id='',
                 src_type='all', dest_type='all',
                 dest_ports=slice(None, None)):
        """
        Mask of source neurons with connections to destination neurons.
        
        Parameters
        ----------
        src_id, dest_id : str
           Module IDs. If no IDs are specified, the IDs stored in
           attributes `A_id` and `B_id` are used in that order.
        src_type : {'all', 'gpot', 'spike'}
           Return a mask over all source neurons ('all'), only
           the graded potential neurons ('gpot'), or only the spiking
           neurons ('spike').
        dest_ports : int or slice
           Only look for source ports with connections to the specified
           destination ports.        
        """

        if src_id == '' and dest_id == '':
            src_id = self.A_id
            dest_id = self.B_id

        self._validate_mod_names(src_id, dest_id)
        if src_type not in ['all', 'gpot', 'spike'] or \
            dest_type not in ['all', 'gpot', 'spike']:
            raise ValueError('invalid neuron type')
        dir = '/'.join((src_id, dest_id))
        if src_type == 'all':
            src_slice = slice(None, None)
        else:
            src_slice = self.idx_translate[src_id][src_type, :]
        if dest_type == 'all':
            dest_slice = dest_ports
        else:                
            dest_slice = self.idx_translate[dest_id][dest_type, dest_ports]
        all_dest_idx = np.arange(self.N(dest_id))[dest_slice]
        result = np.zeros(self.N(src_id, src_type), dtype=bool)
        for k in self._keys_by_dir[dir]:
            result[:] = result+ \
                [np.asarray([bool(np.intersect1d(all_dest_idx, r).size) \
                             for r in self._data[k].rows[src_slice]])]
        return result
        
    def src_idx(self, src_id='', dest_id='',
                src_type='all', dest_type='all',
                dest_ports=slice(None, None)):        
        """
        Indices of source neurons with connections to destination neurons.

        See Also
        --------
        Connectivity.src_mask
        """

        if src_id == '' and dest_id == '':
            src_id = self.A_id
            dest_id = self.B_id
        mask = self.src_mask(src_id, dest_id, src_type, dest_type, dest_ports)    
        if src_type == 'all':            
            return np.arange(self.N(src_id))[mask]
        elif src_type == 'gpot':
            return np.arange(self.N_gpot(src_id))[mask]
        elif src_type == 'spike':
            return np.arange(self.N_spike(src_id))[mask]

    def dest_mask(self, src_id='', dest_id='',
                 src_type='all', dest_type='all',
                 src_ports=slice(None, None)):
        """
        Mask of destination with connections to source neurons.
        
        Parameters
        ----------
        src_id, dest_id : str
           Module IDs. If no IDs are specified, the IDs stored in
           attributes `A_id` and `B_id` are used in that order.
        src_type : {'all', 'gpot', 'spike'}
           Return a mask over all source neurons ('all'), only
           the graded potential neurons ('gpot'), or only the spiking
           neurons ('spike').
        src_ports : int or slice
           Only look for destination ports with connections to the specified
           source ports.        
        """

        if src_id == '' and dest_id == '':
            src_id = self.A_id
            dest_id = self.B_id

        self._validate_mod_names(src_id, dest_id)
        if src_type not in ['all', 'gpot', 'spike'] or \
            dest_type not in ['all', 'gpot', 'spike']:
            raise ValueError('invalid neuron type')
        dir = '/'.join((src_id, dest_id))
        if src_type == 'all':
            src_slice = src_ports
        else:
            src_slice = self.idx_translate[src_id][src_type, src_ports]
        if dest_type == 'all':
            dest_slice = slice(None, None) 
        else:                
            dest_slice = self.idx_translate[dest_id][dest_type, :]
        result = np.zeros(self.N(dest_id), dtype=bool)
        for k in self._keys_by_dir[dir]:
            for r in self._data[k].rows[src_slice]:
                result[r] = True
        return result[dest_slice]

    def dest_idx(self, src_id='', dest_id='',
                src_type='all', dest_type='all',
                src_ports=slice(None, None)):        
        """
        Indices of destination neurons with connections to source neurons.

        See Also
        --------
        Connectivity.src_mask
        """

        if src_id == '' and dest_id == '':
            src_id = self.A_id
            dest_id = self.B_id
        mask = self.dest_mask(src_id, dest_id, src_type, dest_type, src_ports)    
        if dest_type == 'all':
            return np.arange(self.N(dest_id))[mask]
        elif dest_type == 'gpot':
            return np.arange(self.N_gpot(dest_id))[mask]
        elif dest_type == 'spike':
            return np.arange(self.N_spike(dest_id))[mask]
    
    def multapses(self, src_id, src_type, src_idx, dest_id, dest_type,
                  dest_idx):
        """
        Return number of multapses for the specified connection.
        """
        
        assert src_type in ['gpot', 'spike', 'all']
        assert dest_type in ['gpot', 'spike', 'all']
        self._validate_mod_names(src_id, dest_id)
        if src_type == 'all':
            src_idx_new = src_idx
        else:
            src_idx_new = self.idx_translate[src_id][src_type, src_idx]
        if dest_type == 'all':
            dest_idx_new = dest_idx
        else:
            dest_idx_new = self.idx_translate[dest_id][dest_type, dest_idx]
            
        dir = '/'.join((src_id, dest_id))
        count = 0
        for k in self._keys_by_dir[dir]:
            conn, name = k.split('/')[2:]
            conn = int(conn)
            if name == 'conn' and \
                self.get(src_id, src_type, src_idx, dest_id,
                         dest_type, dest_idx, conn, name):
                count += 1
        return count
        
    def get(self, src_id, src_type, src_idx,
            dest_id, dest_type, dest_idx,
            conn=0, param='conn'):
        """
        Retrieve a value in the connectivity class instance.
        """

        assert src_type in ['gpot', 'spike', 'all']
        assert dest_type in ['gpot', 'spike', 'all']
        if src_type == 'all':
            src_idx_new = src_idx
        else:
            src_idx_new = self.idx_translate[src_id][src_type, src_idx]
        if dest_type == 'all':
            dest_idx_new = dest_idx
        else:
            dest_idx_new = self.idx_translate[dest_id][dest_type, dest_idx]    
        return super(Connectivity, self).get(src_id, src_idx_new,
                                             dest_id, dest_idx_new, conn, param)

    def set(self, src_id, src_type, src_idx, dest_id, dest_type, dest_idx,
            conn=0, param='conn', val=1):
        """
        Set a value in the connectivity class instance.
        """
        
        assert src_type in ['gpot', 'spike', 'all']
        assert dest_type in ['gpot', 'spike', 'all']
        if src_type == 'all':
            src_idx_new = src_idx
        else:
            src_idx_new = self.idx_translate[src_id][src_type, src_idx]
        if dest_type == 'all':
            dest_idx_new = dest_idx
        else:
            dest_idx_new = self.idx_translate[dest_id][dest_type, dest_idx]
        return super(Connectivity, self).set(src_id, src_idx_new,
                                             dest_id, dest_idx_new,
                                             conn, param, val=val)
    
    def __repr__(self):
        return super(Connectivity, self).__repr__()+\
          '\nA idx\n'+self.idx_translate[self.A_id].__repr__()+\
          '\n\nB idx\n'+self.idx_translate[self.B_id].__repr__()

class Module(base.BaseModule):
    """
    Processing module.

    This class repeatedly executes a work method until it receives
    a quit message via its control port.

    Parameters
    ----------
    port_data : int
        Port to use when communicating with broker.
    port_ctrl : int
        Port used by broker to control module.
    id : str
        Module identifier. If no identifier is specified, a unique identifier is
        automatically generated.
    device : int
        GPU device to use.
        
    Notes
    -----
    A module instance connected to other module instances contains a list of the
    connectivity objects that describe incoming connects and a list of
    masks that select for the neurons whose data must be transmitted to
    destination modules.    

    """

    def __init__(self, port_data=base.PORT_DATA,
                 port_ctrl=base.PORT_CTRL, id=None, device=None):
        self.device = device
        super(Module, self).__init__(port_data, port_ctrl, id)
                        
    def _init_gpu(self):
        """
        Initialize GPU device.

        Notes
        -----
        Must be called from within the `run()` method, not from within
        `__init__()`.

        """

        if self.device == None:
            self.logger.info('no GPU specified - not initializing ')
        else:
            drv.init()
            try:
                self.gpu_ctx = drv.Device(self.device).make_context()
            except Exception as e:
                self.logger.info('_init_gpu exception: ' + e.message)
            else:
                atexit.register(self.gpu_ctx.pop)
                self.logger.info('GPU initialized')

    def add_conn(self, conn):
        """
        Add the specified connectivity object.

        Parameters
        ----------
        conn : Connectivity
            Connectivity object.

        Notes
        -----
        The module's ID must be one of the two IDs specified in the
        connnectivity object.
        
        """
        
        if not isinstance(conn, Connectivity):
            raise ValueError('invalid connectivity object')        
        super(Module, self).add_conn(conn)
        
    @property
    def N_gpot(self):
        """
        Number of exposed graded-potential neurons.

        Notes
        -----
        Should be overwritten to return the actual number of neurons.

        """

        raise NotImplementedError('N_in_gpot must be implemented')

    @property
    def N_spike(self):
        """
        Number of exposed spiking neurons.

        Notes
        -----
        Must be overwritten to return the actual number of neurons.

        """

        raise NotImplementedError('N_in_spike must be implemented')

    @property
    def N(self):
        """
        Total number of exposed neurons.

        """

        return self.N_gpot+self.N_spike

    def _get_in_data(self, in_gpot_dict, in_spike_dict):
        """
        Get input neuron data from incoming transmission buffer.

        Input neuron data received from other modules is used to populate the
        specified data structures.

        Parameters
        ----------
        in_gpot_dict : dict of numpy.array of float
            Dictionary of graded potential neuron states from other modules.
        in_spike_dict : dict of numpy.array of int
            Dictionary of spiking neuron indices from other modules.
                
        """
        
        self.logger.info('reading input buffer')
        for entry in self._in_data:
            
            # Every received data packet must contain a source module ID,
            # graded potential neuron data, and spiking neuron data:
            if len(entry) != 2:
                self.logger.info('ignoring invalid input data')
            else:
                in_gpot_dict[entry[0]] = entry[1][0]
                in_spike_dict[entry[0]] = entry[1][1]
            
        # Clear input buffer of reading all of its contents:
        self._in_data = []
        
    def _put_out_data(self, out_gpot, out_spike):
        """
        Put specified output neuron data in outgoing transmission buffer.

        Using the indices of the neurons in destination modules that receive
        input from this instance, data extracted from the module's neurons
        is staged for output transmission.

        Parameters
        ----------
        out_gpot : numpy.ndarray of float
            Output neuron states.
        out_spike : numpy.ndarray of int
            Indices of spiking neurons that emitted a spike.
            
        """

        self.logger.info('populating output buffer')

        # Clear output buffer before populating it:
        self._out_data = []
        
        # Use indices of destination neurons to select which neuron
        # values or spikes need to be transmitted to each destination
        # module:
        for out_id in self.out_ids:
            out_idx_gpot = \
                self._conn_dict[out_id].src_idx(self.id, out_id, 'gpot')
            out_idx_spike = \
                self._conn_dict[out_id].src_idx(self.id, out_id, 'spike')

            # Extract neuron data, wrap it in a tuple containing the
            # destination module ID, and stage it for transmission. Notice
            # that since out_spike contains neuron indices, those indices
            # that need to be transmitted can be obtained via a set
            # operation:
            self._out_data.append((out_id,
                                   (np.asarray(out_gpot)[out_idx_gpot],
                                    np.asarray(np.intersect1d(out_spike, out_idx_spike)))))
        
    def run_step(self, in_gpot_dict, in_spike_dict, out_gpot, out_spike):
        """
        Perform a single step of processing.

        Parameters
        ----------
        in_gpot_dict : dict of array_like
            Arrays of input graded potential neuron data for the module to process;
            each key is a source module ID.
        in_spike_dict : dict of array_like
            Arrays of input spiking neuron indices for the module to process;
            each key is a source module ID.
        out_gpot : array_like
            Array of graded potential neuron data to transmit to other modules.
        out_spike : array_like
            Array of output spiking neuron indices to transmit to other modules.

        Notes
        -----
        The index of each array of graded potential neuron data is assumed to
        correspond to the neuron's ID; the arrays of spiking neuron data contain
        the indices of those neurons that have emitted a spike.
        
        """

        self.logger.info('running execution step')
        
    def run(self):        
        """
        Body of process.
        """
                
        # Don't allow keyboard interruption of process:
        self.logger.info('starting')
        with IgnoreKeyboardInterrupt():

            # Initialize environment:
            self._init_net()
            self._init_gpu()

            # Initialize data structures for passing data to and from the
            # run_step method:
            in_gpot_dict = {}
            in_spike_dict = {}
            out_gpot = []
            out_spike = []                    
            while True:

                # Get transmitted input data for processing:
                catch_exception(self._get_in_data, self.logger.info,
                                in_gpot_dict, in_spike_dict)

                # Run the processing step:
                catch_exception(self.run_step, self.logger.info,
                                in_gpot_dict, in_spike_dict,     
                                out_gpot, out_spike)

                # Stage generated output data for transmission to other
                # modules:
                catch_exception(self._put_out_data, self.logger.info,
                                out_gpot, out_spike)

                # Synchronize:
                catch_exception(self._sync, self.logger.info)

        self.logger.info('exiting')
        
class Manager(base.BaseManager):
    """
    Module manager.

    Parameters
    ----------
    port_data : int
        Port to use for communication with modules.
    port_ctrl : int
        Port used to control modules.

    """

    def connect(self, m_A, m_B, conn):
        """
        Connect two module instances with a connectivity object instance.

        Parameters
        ----------
        m_A, m_B : Module
            Module instances to connect
        conn : Connectivity
            Connectivity object.
        
        """

        if not isinstance(conn, Connectivity):
            raise ValueError('invalid type')
        
        # Check whether the numbers of source and destination graded potential
        # neurons and spiking neurons supported by the connectivity object
        # are compatible:
        if not((m_A.N_gpot == conn.N_A_gpot and \
                m_A.N_spike == conn.N_A_spike and \
                m_B.N_gpot == conn.N_B_gpot and \
                m_B.N_spike == conn.N_B_spike) or \
               (m_A.N_gpot == conn.N_B_gpot and \
                m_A.N_spike == conn.N_B_spike and \
                m_B.N_gpot == conn.N_A_gpot and \
                m_B.N_spike == conn.N_A_spike)):
            raise ValueError('modules and connectivity objects are incompatible')

        super(Manager, self).connect(m_A, m_B, conn)

if __name__ == '__main__':

    class MyModule(Module):
        """
        Example of derived module class.
        """

        def __init__(self, N_gpot, N_spike, 
                     port_data=base.PORT_DATA,
                     port_ctrl=base.PORT_CTRL, id=None, device=None):
            super(MyModule, self).__init__(port_data, port_ctrl, id, device)
            self.gpot_data = np.zeros(N_gpot, np.float64)
            self.spike_data = np.zeros(N_spike, int)

        @property 
        def N_gpot(self):
            return len(self.gpot_data)

        @property
        def N_spike(self):
            return len(self.spike_data)

        def run_step(self, in_gpot_dict, in_spike_dict,                  
                     out_gpot, out_spike):
            super(MyModule, self).run_step(in_gpot_dict, in_spike_dict, 
                                           out_gpot, out_spike)

            # Perform some random transformations of the graded potential neuron
            # data:        
            # temp = np.random.randint(0, 5, self.N_in_gpot)
            # for i in in_gpot_dict.keys():
            #     temp += np.random.randint(-1, 1, 1)*in_gpot_dict[i][0]            
            # out_gpot[:] = temp
            out_gpot[:] = np.random.rand(self.N_gpot)
            
            # Randomly select neurons to emit spikes:
            # out_spike[:] = \
            #     sorted(set(np.random.randint(0, self.N_in_spike,
            #                                  np.random.randint(0, self.N_in_spike))))
            out_spike[:] = np.arange(self.N_spike)
            
    logger = base.setup_logger()

    man = Manager()
    man.add_brok()

    N1_gpot = N1_spike = 1
    N2_gpot = N2_spike = 2
    m1 = man.add_mod(MyModule(N1_gpot, N1_spike, 
                              man.port_data, man.port_ctrl))
    m2 = man.add_mod(MyModule(N2_gpot, N2_spike, 
                              man.port_data, man.port_ctrl))
    # m3 = MyModule(N, 'unconnected', man.port_data, man.port_ctrl)
    # man.add_mod(m3)
    # m4 = MyModule(N-2, 'unconnected', man.port_data, man.port_ctrl)
    # man.add_mod(m4)    

    conn1 = Connectivity(N1_gpot, N1_spike, N2_gpot, N2_spike, 1, m1.id, m2.id)
    # c1to2['all',:,'all',:,0,'+'] = \
    #     rand_bin_matrix((N1_gpot+N1_spike, N2_gpot+N2_spike),
    #                     (N1_gpot+N1_spike)*(N2_gpot+N2_spike)/2, int)
    # c1to2[m1.id,'all',:,m2.id,'all',:] = np.ones((N1_gpot+N1_spike,
    #                                               N2_gpot+N2_spike), int)
    conn1[m1.id,'all',:,m2.id,'all',:] = \
        np.ones((N1_gpot+N1_spike,
                 N2_gpot+N2_spike))      
    conn1[m2.id,'all',:,m1.id,'all',:] = \
        np.ones((N2_gpot+N2_spike,
                 N1_gpot+N1_spike))
          
    print conn1
    
    # c3to4 = Connectivity(rand_bin_matrix((N-2, N), N**2/2, int))
    # c4to1 = Connectivity(rand_bin_matrix((N, N-2), N**2/2, int)) 
    # c1to3 = Connectivity(rand_bin_matrix((N, N), N**2/2, int))    
    # c1to2 = Connectivity([[1, 0, 0],
    #                       [0, 0, 0],
    #                       [0, 0, 1]])
    # c2to3 = Connectivity([[1, 0, 1],
    #                       [0, 1, 0],
    #                       [0, 0, 0]])
    # c3to1 = Connectivity([[0, 0, 0],
    #                       [0, 1, 0],
    #                       [0, 0, 1]])



    man.connect(m1, m2, conn1)
    # man.connect(m2, m3, c2to3)
    # man.connect(m1, m3, c1to3)
    # man.connect(m3, m4, c3to4)
    # man.connect(m4, m1, c4to1)
    
    man.start()
    time.sleep(2)
    man.stop()
    logger.info('all done')
