"""This is the Connectivity.py class. This class comprises the connectivity
between neurons (spiking and non-spiking) of two neurokernel modules and the
parameters associated to the synapses.

Known issues
------------
    - This class do not support dynamic modifications yet, which means that it
    is not possible change the mapping between two modules in run-time.

"""
import numpy as np

class Connectivity(object):
    """This module comprises the connectivity between neurons (spiking and
    non-spiking) of two neurokernel modules and the parameters associated to
    the synapses.

    Attributes
    ----------
    map : array_like
        This array represents the connections between neurons in different
        modules and has the following format:

              out1  out2  out3  out4
        in1 |  x  |  x  |     |  x
        in2 |  x  |     |     |
        in3 |     |  x  |     |  x

        where 'x' means connected and blank means not connected.
    mask : array_like
        This mask contains the indices of neurons that will have their states
        transmitted to the other LPU and will be stored at the same GPU of
        the sender module.
    kwparam : dict
        This variable comprises the additional information needed to
        compute the inputs from other modules. If a neuron (n1) in module 1 is
        connected to a neuron (n2) in module 2, it's necessary
        additional information in order to compute the contribution of
        n1 on n2. The information needed is associated to the type of synapse.

    """

    def __init__(self, map, kwparam):
        r"""Connectivity class constructor.

        Parameters
        ----------
        map : array_like
            This array represents the connections between neurons in different
            modules and has the following format:

                  out1  out2  out3  out4
            in1 |  x  |  x  |     |  x
            in2 |  x  |     |     |
            in3 |     |  x  |     |  x

            where 'x' means connected and blank means not connected.
        kwparam : dict
            Parameters associated to each synapse. See the example below.

        Examples
        --------
        In this example it's created a connection between two already
        instantiated modules, m1 and m2.
        >>> import numpy as np
        >>> ...
        >>> map = np.asarray([[0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                              [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                              [1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                              [0, 0, 1, 0, 1, 0, 1, 1, 0, 0]], dtype = np.bool)
        >>> weights = np.random.rand(5,10) * map
        >>> slope = np.random.rand(5,10) * map
        >>> saturation = np.random.rand(5,10) * map
        >>> con = Connectivity(map, {'weights' : weights, 'slope' : slope,
                                     'saturation' : saturation})
        >>> print con.map
        [[False False  True False  True False  True  True False False]
         [ True False False False  True False False  True False False]
         [False False False False  True False  True False False False]
         [ True False False False  True False  True  True False False]
         [False False  True False  True False  True  True False False]]

        Raises
        ------
        IOError
            When you do not provide a numpy.darray with booleans or when you
            provide an array with ndim <> 2.

            Also when the sum of the length of all types are different of the
            number of rows in map.

        See Also
        --------
        neurokernel.Module : Class connected by the connectivity module.
        neurokernel.Manager : Class that manages Modules and Connectivities.

        """
        if type(map) <> np.ndarray and map.dtype <> bool:
            raise IOError, "You must provide a mapping as a numpy.darray with \
                            booleans"
        if map.ndim <> 2:
            raise IOError, "You must provide a 2D numpy.darray"

        if type(kwparam) <> dict:
            raise IOError, "Parameters must be as a dictionary."

        temp_validation = set([v.shape for v in kwparam.values()])
        if len(temp_validation) <> 1 or temp_validation.pop() <> map.shape:
            raise IOError, "Parameters of one type must have the same length."

        if kwparam.values()[0].shape <> map.shape:
            raise IOError, "The sum of the number of spiking and \
            graded-potential parameters must be equal to the number of rows \
            in mapping matrix."

        # Connectivity matrix
        self.map = map.copy()

        # Parameters
        self.kwparam = kwparam.copy()

        # Mask
        self.mask = self.__process_map(map)

    def __process_map(self, map):
        # support vector in the form:
        # | True | False | False | True | True
        # where True or False mean if it is necessary to transmit or not the
        # state of determined neuron
        map_support = self.map.sum(axis = 0).astype(np.bool)
        # The mask is a vector with the indexes of neurons that transmit data.
        output_mask = np.compress(map_support, map_support * \
                                       range(1, len(map_support) + 1)) - 1
        # The compressed matrices will be stored at the same GPU of the
        # receiver module and represent the connectivity between the neurons
        # that are indeed connected. E.g.: if one module has 100 projection
        # neurons, but just 40 neurons are connected to a module, the matrices
        # need to have a shape like (num_receivers, 40) to map the connections.
        # compressed_map = np.compress(map_support, self.map, axis = 1)

        return output_mask #, compressed_map
