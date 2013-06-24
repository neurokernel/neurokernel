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
from tools.misc import rand_bin_matrix

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
    Inter-LPU connectivity with support for graded potential and spiking
    neurons.

    Stores the connectivity between two LPUs as a series of sparse matrices.
    Every entry in an instance of the class has the following indices

    - source neuron type ('gpot', 'spike', or 'all')
    - source neuron ID
    - destination neuron type ('gpot', 'spike', or 'all')
    - destination neuron ID
    - direction ('+' for source to destination, '-' for destination to source)
    - parameter name (the default is 'conn' for simple connectivity)
 
    Each connection may therefore have several parameters; parameters associated
    with nonexistent connections (i.e., those whose 'conn' parameter is set to
    0) should be ignored.
    
    Parameters
    ----------
    N_src_gpot : int
        Number of source graded potential neurons.
    N_src_spike : int
        Number of source spiking neurons.
    N_dest_gpot : int
        Number of destination graded potential neurons.
    N_dest_spike : int
        Number of destination spiking neurons.

    Examples
    --------
    The first connection between spiking neuron 0 in one LPU with graded
    potential 3 in some other LPU can be accessed as
    c['spike',0,'gpot',3,0,'+']. The 'weight' parameter associated with this
    connection can be accessed as c['spike',0,'gpot',3,0,'+','weight']

    """
    
    def __init__(self, N_src_gpot, N_src_spike, N_dest_gpot, N_dest_spike):
        self.N_src_gpot = N_src_gpot
        self.N_src_spike = N_src_spike
        self.N_dest_gpot = N_dest_gpot
        self.N_dest_spike = N_dest_spike
        super(Connectivity, self).__init__(N_src_gpot+N_src_spike,
                                           N_dest_gpot+N_dest_spike)

        # Create index translators to enable use of separate sets of identifiers
        # for graded potential and spiking neurons:
        self.idx_translate = {}
        if self.N_src_gpot == 0:
            self.idx_translate['src'] = \
                IntervalIndex([0, self.N_src_spike], ['spike'])
        elif self.N_src_spike == 0:
            self.idx_translate['src'] = \
                IntervalIndex([0, self.N_src_gpot], ['gpot'])
        else:
            self.idx_translate['src'] = \
                IntervalIndex([0, self.N_src_gpot,
                               self.N_src_gpot+self.N_src_spike],
                              ['gpot', 'spike'])
        if self.N_dest_gpot == 0:
            self.idx_translate['dest'] = \
                IntervalIndex([0, self.N_dest_spike], ['spike'])
        elif self.N_dest_spike == 0:
            self.idx_translate['dest'] = \
                IntervalIndex([0, self.N_dest_gpot], ['gpot'])
        else:
            self.idx_translate['dest'] = \
                IntervalIndex([0, self.N_dest_gpot,
                               self.N_dest_gpot+self.N_dest_spike],
                              ['gpot', 'spike'])

    @property
    def src_mask_gpot(self):
        """
        Mask of source graded potential neurons with connections to destination neurons.
        """

        m_list = [self._data[k][self.idx_translate['src']['gpot', :], :] for k in self._keys_by_dir['+']]
        return np.any(np.sum(m_list).toarray(), axis=1)

    @property
    def src_idx_gpot(self):
        """
        Indices of source graded potential neurons with connections to destination neurons.
        """

        return np.arange(self.N_src_gpot)[self.src_mask_gpot]
                         
    @property
    def src_mask_spike(self):
        """
        Mask of source spiking neurons with connections to destination neurons.
        """

        m_list = [self._data[k][self.idx_translate['src']['spike', :], :] for k in self._keys_by_dir['+']]
        return np.any(np.sum(m_list).toarray(), axis=1)

    @property
    def src_idx_spike(self):
        """
        Indices of source spiking neurons with connections to destination neurons.
        """

        return np.arange(self.N_src_spike)[self.src_mask_spike]
    
    def get(self, src_type, src, dest_type, dest,
            syn=0, dir='+', param='conn'):
        """
        Retrieve a value in the connectivity class instance.
        """

        assert src_type in ['gpot', 'spike', 'all']
        assert dest_type in ['gpot', 'spike', 'all']
        if src_type == 'all':
            t1 = self.idx_translate['src'][src]
        else:
            t1 = self.idx_translate['src'][src_type, src]
        if dest_type == 'all':
            t2 = self.idx_translate['dest'][dest]
        else:
            t2 = self.idx_translate['dest'][dest_type, dest]                    
        return super(Connectivity, self).get(t1, t2, syn, dir, param)

    def set(self, src_type, src, dest_type, dest,
            syn=0, dir='+', param='conn', val=1):
        """
        Set a value in the connectivity class instance.
        """
        
        assert src_type in ['gpot', 'spike', 'all']
        assert dest_type in ['gpot', 'spike', 'all']
        if src_type == 'all':
            t1 = src
        else:
            t1 = self.idx_translate['src'][src_type, src]
        if dest_type == 'all':
            t2 = dest
        else:
            t2 = self.idx_translate['dest'][dest_type, dest]
        return super(Connectivity, self).set(t1, t2, syn, dir, param, val=val)
    
    def __repr__(self):
        return super(Connectivity, self).__repr__()+\
          '\nsrc idx\n'+self.idx_translate['src'].__repr__()+\
          '\n\ndest idx\n'+self.idx_translate['dest'].__repr__()

class Module(base.BaseModule):
    """
    Processing module.

    This class repeatedly executes a work method until it receives
    a quit message via its control port.

    Notes
    -----
    A module instance connected to other module instances contains a list of the
    connectivity objects that describe incoming connects and a list of
    masks that select for the neurons whose data must be transmitted to
    destination modules.    

    """

    def __init__(self, net='unconnected', port_data=base.PORT_DATA,
                                  port_ctrl=base.PORT_CTRL, device=None):
        self.device = device
        super(Module, self).__init__(net, port_data, port_ctrl)
                        
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

    def add_conn(self, conn, conn_type, id):
        """
        Add the specified connectivity object.

        Parameters
        ----------
        conn : Connectivity
            Connectivity object.
        conn_type : {'in', 'out'}
            Connectivity type. 
        id :
            ID of module instance that is being connected via the specified
            object.
        
        """
        
        if not isinstance(conn, base.BaseConnectivity):
            raise ValueError('invalid connectivity object')        
        super(Module, self).add_conn(conn, conn_type, id)
        
    @property
    def N_in_gpot(self):
        """
        Number of graded-potential neurons that receive input.

        Notes
        -----
        Should be overwritten to return the actual number of neurons.

        """

        raise NotImplementedError('N_in_gpot must be implemented')

    @property
    def N_in_spike(self):
        """
        Number of spiking neurons that receive input.

        Notes
        -----
        Must be overwritten to return the actual number of neurons.

        """

        raise NotImplementedError('N_in_spike must be implemented')

    @property
    def N_in(self):
        """
        Total number of neurons that receive input.

        """

        return self.N_in_gpot+self.N_in_spike

    @property
    def N_out_gpot(self):
        """
        Number of graded-potential neurons that emit output.

        Notes
        -----
        Must be overwritten to return the actual number of neurons.

        """

        raise NotImplementedError('N_out_gpot must be implemented')

    @property
    def N_out_spike(self):
        """
        Number of spiking neurons that emit output.

        Notes
        -----
        Must be overwritten to return the actual number of neurons.

        """

        raise NotImplementedError('N_out_spike must be implemented')

    @property
    def N_out(self):
        """
        Total number of neurons that emit output.

        """

        return self.N_out_gpot+self.N_out_spike

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
        self.logger.info('input buffer: '+str(self._in_data))
        
        for entry in self._in_data:
            in_gpot_dict[entry[0]] = entry[1]
            in_spike_dict[entry[0]] = entry[2]
            
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
        for id in self.out_ids:
            out_idx_gpot = self.conn_dict['out'][id].src_idx_gpot
            out_idx_spike = self.conn_dict['out'][id].src_idx_spike
            self.logger.info('out_idx_gpot '+str(out_idx_gpot))
            self.logger.info('out_gpot '+str(out_gpot))            
            self.logger.info('out_idx_spike '+str(out_idx_spike))
            self.logger.info('out_spike '+str(out_spike))
            # Extract neuron data, wrap it in a tuple containing the
            # destination module ID, and stage it for transmission. Notice
            # that since out_spike contains neuron indices, those indices
            # that need to be transmitted can be obtained via a set
            # operation:
            self._out_data.append((id, np.asarray(out_gpot)[out_idx_gpot],
                np.asarray(np.intersect1d(out_spike, out_idx_spike))))
        self.logger.info('output buffer: '+str(self._out_data))
        
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
        with TryExceptionOnSignal(self.quit_sig, Exception, self.id):

            # Don't allow keyboard interruption of process:
            self.logger.info('starting')
            with IgnoreKeyboardInterrupt():

                self._init_net()
                self._init_gpu()
                self.running = True

                # Initialize data structures for passing data to and from the
                # run_step method:
                in_gpot_dict = {}
                in_spike_dict = {}
                out_gpot = []
                out_spike = []                    
                while True:

                    # Get transmitted input data for processing:
                    self._get_in_data(in_gpot_dict, in_spike_dict)
                                        
                    # Run the processing step:
                    self.run_step(in_gpot_dict, in_spike_dict,     
                                  out_gpot, out_spike)

                    # Stage generated output data for transmission to other
                    # modules:
                    self._put_out_data(out_gpot, out_spike)
                                        
                    # Synchronize:
                    self._sync()

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

    def connect(self, m_src, m_dest, conn, dir):
        """
        Connect a source and destination module with a connectivity pattern.

        Parameters
        ----------
        m_src : Module
            Source module.
        m_dest : Module
            Destination module
        conn : Connectivity
            Connectivity object.
        dir : {'+','-','='}
           Connectivity direction; '+' denotes a connection from `m_src` to
           `m_dest`; '-' denotes a connection from `m_dest` to `m_src`; '='
           denotes connections in both directions.
        
        """

        # Check whether the numbers of source and destination graded potential
        # neurons and spiking neurons supported by the connectivity object
        # are compatible
        if m_src.N_out_gpot != conn.N_src_gpot or \
            m_dest.N_in_gpot != conn.N_dest_gpot or \
            m_src.N_out_spike != conn.N_src_spike or \
            m_dest.N_in_spike != conn.N_dest_spike:
            raise ValueError('modules and connectivity objects are incompatible')
        
        super(Manager, self).connect(m_src, m_dest, conn, dir)

if __name__ == '__main__':

    class MyModule(Module):
        """
        Example of derived module class.
        """

        def __init__(self, N_gpot, N_spike, net='unconnected',
                     port_data=base.PORT_DATA,
                     port_ctrl=base.PORT_CTRL, device=None):
            super(MyModule, self).__init__(net, port_data, port_ctrl, device)
            self.gpot_data = np.zeros(N_gpot, np.float64)
            self.spike_data = np.zeros(N_spike, int)

        @property 
        def N_in_gpot(self):
            return len(self.gpot_data)

        @property
        def N_in_spike(self):
            return len(self.spike_data)

        @property 
        def N_out_gpot(self):
            return len(self.gpot_data)

        @property
        def N_out_spike(self):
            return len(self.spike_data)

        def run_step(self, in_gpot_dict, in_spike_dict,                  
                     out_gpot, out_spike):
            super(MyModule, self).run_step(in_gpot_dict, in_spike_dict, 
                                           out_gpot, out_spike)

            # Perform some random transformations of the graded potential neuron
            # data:        
            temp = np.random.randint(0, 5, self.N_in_gpot)
            for i in in_gpot_dict.keys():
                temp += np.random.randint(-1, 1, 1)*in_gpot_dict[i][0]            
            out_gpot[:] = temp

            # Randomly select neurons to emit spikes:
            out_spike[:] = \
                sorted(set(np.random.randint(0, self.N_in_spike,
                                             np.random.randint(0, self.N_in_spike))))
            
    logger = base.setup_logger()

    man = Manager()
    man.add_brok()

    N1_gpot = N1_spike = 2
    N2_gpot = N2_spike = 4
    m1 = man.add_mod(MyModule(N1_gpot, N1_spike, 'none',
                              man.port_data, man.port_ctrl))
    m2 = man.add_mod(MyModule(N2_gpot, N2_spike, 'none',
                              man.port_data, man.port_ctrl))
    # m3 = MyModule(N, 'unconnected', man.port_data, man.port_ctrl)
    # man.add_mod(m3)
    # m4 = MyModule(N-2, 'unconnected', man.port_data, man.port_ctrl)
    # man.add_mod(m4)    

    c1to2 = Connectivity(N1_gpot, N1_spike, N2_gpot, N2_spike)
    # c1to2['all',:,'all',:,0,'+'] = \
    #     rand_bin_matrix((N1_gpot+N1_spike, N2_gpot+N2_spike),
    #                     (N1_gpot+N1_spike)*(N2_gpot+N2_spike)/2, int)
    c1to2['all',:,'all',:,0,'+'] = np.ones((N1_gpot+N1_spike,
                                            N2_gpot+N2_spike), int)
    print c1to2
    
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



    man.connect(m1, m2, c1to2, '+')
    # man.connect(m2, m3, c2to3)
    # man.connect(m1, m3, c1to3)
    # man.connect(m3, m4, c3to4)
    # man.connect(m4, m1, c4to1)
    
    man.start()
    time.sleep(2)
    man.stop()
    logger.info('all done')
