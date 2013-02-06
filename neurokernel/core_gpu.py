#!/usr/bin/env python

"""
Core Neurokernel classes that use the GPU.


"""

import atexit
import numpy as np
import twiggy
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import bidict

import tools.autoinit
from tools.autoinit import curr_gpu, switch_gpu
import core

class Connectivity(core.BaseConnectivity):
    """
    Synaptic connectivity between modules.

    Describes the synaptic connections and associated parameters
    between neurons the neurons in two Neurokernel modules.

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
    out_ind : numpy.ndarray of int
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
    def out_ind(self):
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

        # Dictionaries that maps destination module IDs to arrays
        # containing the IDs of the neurons in the destination module
        # that receive input:
        self.out_gpot_ind_dict = {}
        self.out_spike_ind_dict = {}

        # Dictionaries that map source module IDs to GPU arrays
        # containing states of those modules' neurons that are
        # transmitted to this module instance:
        self.in_gpot_gpu_dict = {}
        self.in_spike_gpu_dict = {}
        self.in_spike_count_dict = {}

    @property
    def out_gpot_ids(self):
        """
        IDs of destination modules containing graded-potential neurons.
        """

        return self.out_gpot_ind_dict.keys()

    @property
    def out_spike_ids(self):
        """
        IDs of destination modules containing spiking neurons.
        """

        return self.out_spike_ind_dict.keys()


    def _init_gpu(self):
        """
        Initialize GPU device.

        Notes
        -----
        Must be called from within the `run()` method, not from within
        `__init__()`.

        """

        drv.init()
        self.gpu_ctx = drv.Device(self.device).make_context()
        atexit.register(ctx.pop)

    @property
    def N_in_gpot(self):
        """
        Number of graded-potential neurons that receive input.

        Notes
        -----
        Should be overwritten to return the actual number of neurons.

        """

        return 0

    @property
    def N_in_spike(self):
        """
        Number of spiking neurons that receive input.

        Notes
        -----
        Should be overwritten to return the actual number of neurons.

        """

        return 0

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
        Should be overwritten to return the actual number of neurons.

        """

        return 0

    @property
    def N_out_spike(self):
        """
        Number of spiking neurons that emit output.

        Notes
        -----
        Should be overwritten to return the actual number of neurons.

        """

        return 0

    @property
    def N_out(self):
        """
        Total number of neurons that emit output.

        """

        return self.N_out_gpot+self.N_out_spike

    def _extract_out_data(self):
        """
        Extract specified output neuron data to ship to destination modules.
        """

        # Use indices of destination neurons to select which neuron
        # values or spikes need to be transmitted to each destination
        # module:
        for id in self.out_id_list:
            try:
                gpot_ind = self.out_gpot_ind_dict[id]

                # unfinished
            except:
                pass
        if len(out_gpot_list) != len(out_spike_list):
            raise ValueError('number of graded potential and spiking '
                             'neuron arrays must be equivalent')
        for out_gpot, out_spike in zip(out_gpot_list, out_spike_list):
            pass

    def run_step(self, in_gpot_dict, in_spike_count_dict, in_spike_dict, 
                 out_gpot_gpu, out_spike_count, out_spike_gpu):
        pass
    
    def run(self):
        with TryExceptionOnSignal(self.quit_sig, Exception, self.id):

            # Don't allow keyboard interruption of process:
            self.logger.info('starting')
            with IgnoreKeyboardInterrupt():

                self._init_net()
                self._init_gpu()
                self.running = True
                while True:

                    # Run the processing step:
                    self.run_step()

                    # Synchronize:
                    self._sync()

            self.logger.info('exiting')


class Manager(core.BaseManager):

    def connect(self, m_src, m_dest, conn):

        # Check whether the numbers of source and destination neurons
        # supported by the connectivity object are compatible with the
        # module instances being connected:
        if m_src.N_out != conn.N_in or m_dest.N_in != conn.N_out:
            raise ValueError('modules and connectivity objects are incompatible')

        # Provide an array listing to the source module that lists the
        # indices of those output neurons that project to the
        # destination module:
        m_src.out_spike_ind_dict[m_dest.id] = conn.out_ind

        # Switch to the appropriate context to allocate GPU arrays for
        # incoming neuron state and spike data:
        last_gpu = curr_gpu
        switch_to(m_dest.gpu)
        m_dest.in_gpot_gpu_dict[m_src.id] = \
            gpuarray.zeros(m_src.N_out_gpot, np.double)
        m_dest.in_spike_gpu_dict[m_src.id] = \
            gpuarray.zeros(m_src.N_out_spike, np.int32)
        m_dest.in_spike_count_dict[m_src.id] = 0
        switch_to(last_gpu)

        super(Manager, self).__init__(m_src, m_dest, conn)
