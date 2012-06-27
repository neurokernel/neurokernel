"""This is the Connectivity.py class. This class comprises the connectivity
between neurons (spiking and non-spiking) of two neurokernel modules and the
parameters associated to the synapses.

Known issues
------------
    - This class do not support dynamic modifications yet, which means that it
    is not possible change the mapping between two modules in run-time.

"""
import numpy as np

class Connectivity:
    """This module comprises the connectivity between neurons (spiking and
    non-spiking) of two neurokernel modules and the parameters associated to
    the synapses.

    Attributes
    ----------
    non_map : array_like
        This array represents the connections between neurons in different
        modules and has the following format:

              out1  out2  out3  out4
        in1 |  x  |  x  |     |  x
        in2 |  x  |     |     |
        in3 |     |  x  |     |  x

        where 'x' means connected and blank means not connected.
    output_mask : array_like
        This mask contains the indices of neurons that will have their states
        transmitted to the other LPU and will be stored at the same GPU of
        the sender module.
    compressed_map : array_like
        The compressed matrices will be stored at the same GPU of the
        receiver module and represent the connectivity between the neurons
        that are indeed connected. E.g.: if one module has 100 projection
        neurons, but just 40 neurons are connected to a module, the matrices
        need to have a shape like (num_receivers, 40) to map the connections.
    param_per_type : array_like
        Parameter per type comprises the additional information needed to
        compute the inputs from other modules. If a neuron (n1) in module 1 is
        connected to a type T neuron (nT2) in module 2, it's necessary
        additional information in order to compute the contribution of
        n1 on nT2.

    See also
    --------
    class Parameter

    """

    def __init__(self, map, param_per_type):
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
        param_per_type : array_like
            For each type of destination neuron, it is necessary to specify a
            set of parameters associated to it, e.g.:
            >>> ...
            >>>
            >>> type1 = Parameters(Weight(.2, .3), Slope(.6, .3))
            >>> type2 = Parameters(Weight(.8,), Slope(.7))
            >>> conn = Connectivity(map, [type1, type2])

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

        if type(param_per_type) <> np.ndarray and \
            param_per_type.dtype <> Parameters:
            raise IOError, "Parameters must be provided  by type as shown in \
                            documentation."

        if len(map) <> sum([len(x) for x in param_per_type]):
            raise IOError, "the sum of the length of all types are different \
                            of the number of rows in map"

        self.map = map
        self.param_per_type = param_per_type

        # support vector in the form:
        # | True | False | False | True | True
        # where True or False mean if it is necessary to transmit or not the
        # state of determined neuron
        map_support = self.map.sum(axis = 0).astype(np.bool)
        # The mask is a vector with the indexes of neurons that transmit data.
        self.output_mask = np.compress(map_support, map_support * \
                                       range(1, len(map_support) + 1)) - 1
        self.compressed_map = np.compress(map_support, self.map, axis = 1)

    def connect(self, proj_module, input_module):
        r"""Connects the projection neurons of the source module to the input
        neurons of the destination module. In order to do that, this method
        saves a mask in the source module memory to be applied on the output
        vector and provide itself to the destination module.

        """
        # put the mask inside the source
        proj_module.in_conn.append(self)
        input_module.output.append()

        # put the compressed matrix inside the destination
        return 0

    # Gets the output signal in the form (num_inputs, 1) and
    def get_output(self, module):
        return module.neurons.V.get() * self.non_map

class Parameter:
    """
    Interface representing each element of the composition of parameters.
    The structure is based on the Composite pattern in [1].

    References
    ----------
    .. [1] ErichGamma, RichardHelm, RalphJohnson, and JohnVlissides, Design
       Patterns: Elements of Reusable Object-Oriented Software, 1994

    """

    def __init__(self):
        self.__values = []

    def __len__(self):
        return self.__values.len()

    def __getitem__(self, index):
        return self.__values[index]

class Parameters(Parameter):
    def __init__(self, elements):
        self.__elements = elements

    def add(self, x):
        self.__elements.append(x)

    def remove(self, x):
        self.__elements.remove(x)

    def __len__(self):
        return self.__elements.len()

    def __getitem__(self, index):
        return self.__elements[index]

class Alpha(Parameter):
    def __init__(self, values):
        self.__values = values

class Weight(Parameter):
    def __init__(self, values):
        self.__values = values

class Slope(Parameter):
    def __init__(self, values):
        self.__values = values

class Threshold(Parameter):
    def __init__(self, values):
        self.__values = values

class Saturation(Parameter):
    def __init__(self, values):
        self.__values = values

def main():
    m = np.asarray([[0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0, 1, 1, 0, 0]], dtype = np.bool)
    m = Connectivity(m)
    m.get_output(np.asarray([[1.4, 3.2, 1.7]]))

if __name__ == '__main__':

    # number of neurons per type that will be multiplied by 15
    # average number of synapses per neuron
    # parameters = 768, 6, 1e-4, 4608, 0, 4608, 0, 1
    main()
