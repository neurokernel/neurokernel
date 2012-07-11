"""
Intermodule synaptic connectivity.

Notes
-----
Dynamic modification of synapses parameters not currently supported.

"""

import numpy as np

class Connectivity(object):
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
        neurokernel.Module : Class connected by the Connectivity class
        neurokernel.Manager : Class that manages Module and Connectivity class instances.

        """

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
    def conn(self):
        """
        Active synapses.
        """

        return self._conn

    @property
    def out_indices(self):
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

