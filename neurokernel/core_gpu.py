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

import core

class Connectivity(core.BaseConnectivity):
    """
    Synaptic connectivity between modules.

    Describes the synaptic connections and associated parameters
    between neurons the neurons in two Neurokernel modules.

    Attributes
    ----------
    conn : array_like of bool
        Synaptic connectivity. Has the following format:

              out1  out2  out3  out4
        in1 |  x  |  x  |     |  x
        in2 |  x  |     |     |
        in3 |     |  x  |     |  x

        where 'x' is a connection denoted by a nonzero value.
    params : dict
        Parameters associated with synapses. Each key in the
        dictionary is a parameter name; the associated matrix contains
        the parameter values.

    Notes
    -----
    Dynamic modification of synapse parameters not currently supported.

    """

    def __init__(self, conn, **params):
        """
        Connectivity class constructor.

        Parameters
        ----------
        conn : array_like
            This array represents the connections between neurons in different
            modules and has the following format:

                   in1   in2   in3   in4
            out1 |  x  |  x  |     |  x
            out2 |  x  |     |     |
            out3 |     |  x  |     |  x

            where 'x' means connected and blank means not connected.
        params : dict
            Synaptic parameters. See the example below.

        Attributes
        ----------
        compressed : 2D numpy.ndarray of bool
        conn : 2D numpy.ndarray of bool
            Connectivity matrix.
        out_ind : numpy.ndarray of int
            Indices of source neurons with output connections.
        param_names : list of str
            List of parameter names; each matrix of parameters
            identified by `name` in an object instance `X` can be 
            accessed as `X['name']`
        
        Examples
        --------
        >>> import numpy as np
        >>> ...
        >>> conn = np.asarray([[0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                               [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                               [1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                               [0, 0, 1, 0, 1, 0, 1, 1, 0, 0]], dtype = np.bool)
        >>> weights = np.random.rand(5,10)*conn
        >>> slope = np.random.rand(5,10)*conn
        >>> saturation = np.random.rand(5,10)*conn
        >>> c = Connectivity(map, weights=weights, slope=slope,
                                     saturation=saturation)
        >>> print c.conn
        [[False False  True False  True False  True  True False False]
         [ True False False False  True False False  True False False]
         [False False False False  True False  True False False False]
         [ True False False False  True False  True  True False False]
         [False False  True False  True False  True  True False False]]

        Notes
        -----
        All parameter matrices must have the same dimensions and may
        only specify non-zero entries for active synapses.

        See Also
        --------
        Module : Class connected by the Connectivity class
        Manager : Class that manages Module and Connectivity class instances.

        """

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

        # Find the source neurons that have output connections:
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

        # Discard the columns with no output connections:
        return np.compress(self._out_mask, self._conn, axis=1)

class Module(core.BaseModule):
    """
    GPU-based processing module.

    This class repeatedly executes a work method until it receives 
    a quit message via its control port.
    """
    
    def __init__(self, net='unconnected', port_data=core.PORT_DATA,
                 port_ctrl=core.PORT_CTRL, device=0):        
        self.device = device
        super(Module, self).__init__(net, port_data, port_ctrl)

        # Dictionaries that maps destination module IDs to arrays
        # containing lists of 
        self.out_gpot_inds = bidict.bidict()
        self.out_spike_inds = bidict.bidict()
        
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

    def _extract_out_data(self, out_gpot_list, out_spike_list):
        """
        Extract specified output neuron data to ship to destination modules.
        """

        if len(out_gpot_list) != len(out_spike_list):
            raise ValueError('number of graded potential and spiking '
                             'neuron arrays must be equivalent')
        for out_gpot, out_spike in zip(out_gpot_list, out_spike_list):
            
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
        super(Manager, self).__init__(m_src, m_dest, conn)

        # Provide an array listing to the source module that lists
        # which of its output neurons project to the destination
        # module:

        
