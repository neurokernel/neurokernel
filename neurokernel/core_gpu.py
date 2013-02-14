#!/usr/bin/env python

"""
Core Neurokernel classes that use the GPU.


"""

import atexit
import numpy as np
import time

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import twiggy
import bidict

import core
from ctx_managers import IgnoreKeyboardInterrupt, OnKeyboardInterrupt, \
     ExceptionOnSignal, TryExceptionOnSignal
#import tools.autoinit
#from tools.autoinit import curr_gpu, switch_gpu
from neurokernel.tools.misc_utils import rand_bin_matrix

class Connectivity(core.BaseConnectivity):
    """
    Synaptic connectivity between modules.

    Describes the synaptic connections and associated parameters
    between neurons in two Neurokernel modules.

    Parameters
    ----------
    conn : array_like
        Synaptic connectivity. Has the following format:

               in1   in2   in3   in4
             +-----+-----+-----+-----
        out1 |  x  |  x  |     |  x
        out2 |  x  |     |     |
        out3 |     |  x  |     |  x

        where 'x' is a connection denoted by a nonzero value.
    params : dict
        Parameters associated with synapses. Each key in the
        dictionary is a parameter name; the associated matrix contains
        the parameter values.

    Attributes
    ----------
    N_in : int
        Number of supported input neurons.
    N_out : int
        Number of supported output neurons.
    compressed : 2D numpy.ndarray of bool
        Connectivity matrix with connectionless columns removed.
    conn : 2D numpy.ndarray of bool
        Connectivity matrix.
    out_idx : numpy.ndarray of int
        Indices of input neurons with output connections.
    param_names : list of str
        List of parameter names; each matrix of parameters
        identified by `name` in an object instance `X` can be
        accessed as `X[name]`; if a parameter is not defined 
        for a particular synapse, it's value is set to None.

    Examples
    --------
    >>> import numpy as np
    >>> conn = np.asarray([[0, 0, 1],
                           [1, 0, 0],
                           [0, 0, 1]])
    >>> weights = np.random.rand(3, 3)*conn
    >>> c = Connectivity(map, weights=weights)
    connectivity:
    [[0 0 1]
     [1 0 0]
     [0 0 1]]
    weights:
    [[ 0.          0.          0.05855326]
     [ 0.80546794  0.          0.        ]
     [ 0.          0.          0.16805278]]

    Notes
    -----
    All parameter matrices must have the same dimensions and may
    only specify non-zero entries for active synapses.

    Dynamic modification of synapse parameters not currently supported.

    See Also
    --------
    Module : Class connected by the Connectivity class
    Manager : Class that manages Module and Connectivity class instances.

    """

    def __init__(self, conn, **params):
        super(Connectivity, self).__init__()

        if np.ndim(conn) != 2:
            raise ValueError('connectivity matrix must be 2D')
        self._conn = np.array(conn, dtype=bool, copy=True)

        param_shapes = set([self._conn.shape]+[np.shape(p) for p in params.values()])
        if len(param_shapes) > 1:
            raise ValueError('all parameter matrices must have the same shape')

        # Nonzero values in the various parameter matrices may not
        # appear at coordinates that do not correspond to active synapses:
        for p in params.values():
            if np.any((np.asarray(self._conn, int)-np.asarray(p>0, int))<0):
                raise ValueError('parameter may only be specified for active synapses')

        # Save parameters:
        self._params = params.copy()

        # Find the input neurons that have output connections:
        self._out_mask = np.any(self.conn, axis=0)

    def __getitem__(self, p):
        return self._params[p]

    @property
    def param_names(self):
        """
        Parameter names.
        """
        return self._params.keys()

    @property
    def conn(self):
        """
        Active synapses.
        """

        return self._conn

    def __repr__(self):
        result =  'connectivity:\n'+str(self._conn.astype(int))
        for k in self._params.keys():
            result += '\n'+k+':\n'+str(self._params[k])
        return result

    @property
    def N_in(self):
        """
        Number of input neurons supported.
        """

        return self._conn.shape[1]

    @property
    def N_out(self):
        """
        Number of output neurons supported.
        """

        return self._conn.shape[0]

    @property
    def out_idx(self):
        """
        Return indices of source neurons with output connections.
        """

        return np.arange(self._conn.shape[1])[self._out_mask]

    @property
    def compressed(self):
        """
        Return connectivity matrix with connectionless columns discarded.
        """

        return np.compress(self._out_mask, self._conn, axis=1)

class Module(core.BaseModule):
    """
    GPU-based processing module.

    This class repeatedly executes a work method until it receives
    a quit message via its control port.

    Notes
    -----
    When a module instance is connected to another module instance,

    """

    def __init__(self, net='unconnected', port_data=core.PORT_DATA,
                 port_ctrl=core.PORT_CTRL, device=0):
        self.device = device
        super(Module, self).__init__(net, port_data, port_ctrl)

        # Dictionaries that map destination module IDs to arrays containing the
        # IDs of the neurons in the destination module that receive input:
        self.out_gpot_idx_dict = {}
        self.out_spike_idx_dict = {}

        # Dictionaries that map source module IDs to arrays containing states of
        # those modules' neurons that are transmitted to this module instance:
        self.in_gpot_dict = {}
        self.in_spike_dict = {}
        self.in_spike_count_dict = {}

        # Connectivity objects:
        self.in_gpot_conn_dict = {}
        
    @property
    def out_gpot_ids(self):
        """
        IDs of destination modules containing graded-potential neurons.
        """

        return self.out_gpot_idx_dict.keys()

    @property
    def out_spike_ids(self):
        """
        IDs of destination modules containing spiking neurons.
        """

        return self.out_spike_idx_dict.keys()


    def _init_gpu(self):
        """
        Initialize GPU device.

        Notes
        -----
        Must be called from within the `run()` method, not from within
        `__init__()`.

        """

        drv.init()
        try:
            self.gpu_ctx = drv.Device(self.device).make_context()
        except Exception as e:
            self.logger.info('_init_gpu exception: ' + e.message)
        else:
            atexit.register(self.gpu_ctx.pop)
            self.logger.info('GPU initialized')

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

    def _get_in_data(self, in_gpot_dict, in_spike_count_dict, in_spike_dict):
        """
        Get input neuron data from incoming transmission buffer.

        Input neuron data received from other modules is used to populate the
        specified data structures.

        Notes
        -----
        This currently only manipulates data from graded potential neurons.
        
        """
        self.logger.info('reading input buffer')
        
        for entry in self.in_data:
            in_gpot_dict[entry[0]] = entry[1]

        # Clear input buffer of reading all of its contents:
        self.in_data = []
        
    def _put_out_data(self, out_gpot, out_spike_count, out_spike):
        """
        Put specified output neuron data on outgoing transmission buffer.

        Using the indices of the neurons in destination modules that receive
        input from this instance, data is extracted from the module's neurons
        and staged for output transmission.
        
        Notes
        -----
        This currently only manipulates data from graded potential neurons.
        
        """

        self.logger.info('populating output buffer')
        
        # Clear output buffer before populating it:
        self.out_data = []
        
        # Use indices of destination neurons to select which neuron
        # values or spikes need to be transmitted to each destination
        # module:        
        for id in self.out_ids:
            try:
                out_gpot_idx = self.out_gpot_idx_dict[id]
                #out_spike_idx = self.out_spike_idx_dict[id]
            except:
                pass
            else:
                # Extract neuron data, wrap it in a tuple containing the
                # destination module ID, and stage it for transmission:
                self.out_data.append((id, np.asarray(out_gpot)[out_gpot_idx]))

    def run_step(self, in_gpot_dict, in_spike_count_dict, in_spike_dict, 
                 out_gpot, out_spike_count, out_spike):
        """
        Perform a single step of processing.

        Parameters
        ----------
        in_gpot_dict : dict of array_like
            Input graded potential neuron data; each key is a source module ID.
        in_spike_count_dict : dict of int
            Number of input neurons from each source module that have emitted a spike.
        in_spike_dict : dict of array_like
            Arrays of spiking neuron indices from each source module ID.
        out_gpot : array_like
            Graded potential neuron data to transmit to other modules.
        out_spike_count : array_like
            Number of neurons that have emitted a spike.
        out_spike : array_like
            Array of spiking neuron indices.
            
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
                in_spike_count_dict = {}
                in_spike_dict = {}
                out_gpot = []
                out_spike_count = [0]
                out_spike = []                    
                while True:

                    # Get transmitted input data for processing:
                    self._get_in_data(in_gpot_dict, in_spike_count_dict,
                                      in_spike_dict)
                    
                    # Run the processing step:
                    self.run_step(in_gpot_dict, in_spike_count_dict,
                                  in_spike_dict, 
                                  out_gpot, out_spike_count, out_spike)

                    # Stage generated output data for transmission to other
                    # modules:
                    self._put_out_data(out_gpot, out_spike_count, out_spike)
                                        
                    # Synchronize:
                    self._sync()

            self.logger.info('exiting')

class Manager(core.BaseManager):
    """
    Module manager.

    Parameters
    ----------
    port_data : int
        Port to use for communication with modules.
    port_ctrl : int
        Port used to control modules.

    """

    def connect(self, m_src, m_dest, conn):
        """

        Notes
        -----
        This currently only sets up connections for graded potential neurons.     
        
        """
        
        # Check whether the numbers of source and destination neurons
        # supported by the connectivity object are compatible with the
        # module instances being connected:
        if m_src.N_out != conn.N_in or m_dest.N_in != conn.N_out:
            raise ValueError('modules and connectivity objects are incompatible')

        # Provide an array listing to the source module that lists the
        # indices of those output neurons that project to the
        # destination module:
        m_src.out_gpot_idx_dict[m_dest.id] = conn.out_idx

        # Save the connectivity objects in the destination module:
        m_dest.in_gpot_conn_dict[m_src.id] = conn
        
        # Switch to the appropriate context to allocate GPU arrays for
        # incoming neuron state and spike data:
        # last_gpu = curr_gpu
        # switch_to(m_dest.gpu)
        # m_dest.in_gpot_gpu_dict[m_src.id] = \
        #     gpuarray.zeros(m_src.N_out_gpot, np.double)
        # m_dest.in_spike_gpu_dict[m_src.id] = \
        #     gpuarray.zeros(m_src.N_out_spike, np.int32)
        # m_dest.in_spike_count_dict[m_src.id] = 0
        # switch_to(last_gpu)

        super(Manager, self).connect(m_src, m_dest, conn)

class MyModule(Module):
    """
    Example module.
    """

    def __init__(self, N, net='unconnected', port_data=core.PORT_DATA,
                 port_ctrl=core.PORT_CTRL, device=0):
        super(MyModule, self).__init__(net, port_data, port_ctrl, device)
        self.N = N

        self.gpot_data = np.zeros(self.N, np.float64)
        self.spike_data = np.zeros(self.N, int)
        
    @property 
    def N_in_gpot(self):
        return len(self.gpot_data)

    @property
    def N_in_spike(self):
        return 0

    @property 
    def N_out_gpot(self):
        return len(self.gpot_data)

    @property
    def N_out_spike(self):
        return 0
    
    def run_step(self, in_gpot_dict, in_spike_count_dict,
                 in_spike_dict, 
                 out_gpot, out_spike_count, out_spike):
        super(MyModule, self).run_step(in_gpot_dict, in_spike_count_dict,
                                       in_spike_dict, 
                                       out_gpot, out_spike_count, out_spike)
        temp = np.random.randint(0, 5, self.N_in_gpot)
        for i in in_gpot_dict.keys():
            temp += np.random.randint(-1, 1, 1)*in_gpot_dict[i][0]            
        out_gpot[:] = temp
        #        out_gpot[:] = np.random.randint(0, 5, self.N_in_gpot)
        
if __name__ == '__main__':
    logger = core.setup_logger()

    man = Manager()
    man.add_brok()

    N = 5
    m1 = MyModule(N, 'unconnected', man.port_data, man.port_ctrl)
    man.add_mod(m1)
    m2 = MyModule(N-2, 'unconnected', man.port_data, man.port_ctrl)
    man.add_mod(m2)
    m3 = MyModule(N, 'unconnected', man.port_data, man.port_ctrl)
    man.add_mod(m3)
    m4 = MyModule(N-2, 'unconnected', man.port_data, man.port_ctrl)
    man.add_mod(m4)    

    c1to2 = Connectivity(rand_bin_matrix((N-2, N), N**2/2, int))
    c2to3 = Connectivity(rand_bin_matrix((N, N-2), N**2/2, int))
    c3to4 = Connectivity(rand_bin_matrix((N-2, N), N**2/2, int))
    c4to1 = Connectivity(rand_bin_matrix((N, N-2), N**2/2, int)) 
    c1to3 = Connectivity(rand_bin_matrix((N, N), N**2/2, int))    
    # c1to2 = Connectivity([[1, 0, 0],
    #                       [0, 0, 0],
    #                       [0, 0, 1]])
    # c2to3 = Connectivity([[1, 0, 1],
    #                       [0, 1, 0],
    #                       [0, 0, 0]])
    # c3to1 = Connectivity([[0, 0, 0],
    #                       [0, 1, 0],
    #                       [0, 0, 1]])



    man.connect(m1, m2, c1to2)
    man.connect(m2, m3, c2to3)
    man.connect(m1, m3, c1to3)
    man.connect(m3, m4, c3to4)
    man.connect(m4, m1, c4to1)
    
    man.start()
    time.sleep(3)
    man.stop()
    logger.info('all done')
