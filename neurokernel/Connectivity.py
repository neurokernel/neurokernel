"""This is the Connectivity.py module. This module comprises the connectivity
between neurons (spiking and non-spiking) of two neurokernel modules and the
parameters associated to the synapses.

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
        compute the inputs from other modules. The structure is based on the
        Composite pattern in [1]. E.g.: If a neuron (n1) in module 1 is
        connected to a type T neuron (nT2) in module 2, it's necessary
        additional information in order to compute the contribution of n1 on
        nT2.

    References
    ----------
    .. [1] ErichGamma, RichardHelm, RalphJohnson, and JohnVlissides, Design
       Patterns: Elements of Reusable Object-Oriented Software, 1994

    """

    def __init__(self, map, param_per_type):
        r"""Connectivity class contructor.

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
            >>> type1 = ParametersComposition(Weight(.2, .3), Slope(.6, .3))
            >>> type2 = ParametersComposition(Weight(.8,), Slope(.7))
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
        if map.dtype <> bool:
            raise IOError, "You must provide a mapping as a numpy.darray with \
                            booleans"
        if map.ndim <> 2:
            raise IOError, "You must provide a 2D numpy.darray"

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
        self.output_mask = np.compress(map_support, map_support * \
                                       range(1, len(map_support) + 1)) - 1
        self.compressed_map = np.compress(map_support, self.map, axis = 1)

    def connect(self, proj_module, input_module):
        r"""Connects the projection neurons of the source module to the input
        neurons of the destination module. In order to do that, this method
        saves a mask in the source module memory to be applied on the output
        vector and a compressed map in the destination module memory

        Parameters
        ----------
        proj_module : neurokernel.Module
            Source Module.
        input_module : neurokernel.Module
            Destination Module

        Raises
        ------
        BadException
            Because you shouldn't have done that.

        Notes
        -----
        Notes about the implementation algorithm (if needed).

        This can have multiple paragraphs.

        You may include some math:

        .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

        And even use a greek symbol like :math:`omega` inline.

        Examples
        --------
        These are written in doctest format, and should illustrate how to
        use the function.

        >>> a=[1,2,3]
        >>> print [x + 3 for x in a]
        [4, 5, 6]
        >>> print "a\n\nb"
        a
        b

        """
        # put the mask inside the source
        # put the compressed matrix inside the destination
        return 0

    def disconnect(self):
        r"""A one-line summary that does not use variable names or the
        function name.

        Several sentences providing an extended description. Refer to
        variables using back-ticks, e.g. `var`.

        Parameters
        ----------
        var1 : array_like
            Array_like means all those objects -- lists, nested lists, etc. --
            that can be converted to an array.  We can also refer to
            variables like `var1`.
        var2 : int
            The type above can either refer to an actual Python type
            (e.g. ``int``), or describe the type of the variable in more
            detail, e.g. ``(N,) ndarray`` or ``array_like``.
        Long_variable_name : {'hi', 'ho'}, optional
            Choices in brackets, default first when optional.

        Returns
        -------
        describe : type
            Explanation
        output : type
            Explanation
        tuple : type
            Explanation
        items : type
            even more explaining

        Other Parameters
        ----------------
        only_seldom_used_keywords : type
            Explanation
        common_parameters_listed_above : type
            Explanation

        Raises
        ------
        BadException
            Because you shouldn't have done that.

        See Also
        --------
        otherfunc : relationship (optional)
        newfunc : Relationship (optional), which could be fairly long, in which
                  case the line wraps here.
        thirdfunc, fourthfunc, fifthfunc

        Notes
        -----
        Notes about the implementation algorithm (if needed).

        This can have multiple paragraphs.

        You may include some math:

        .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

        And even use a greek symbol like :math:`omega` inline.

        References
        ----------
        Cite the relevant literature, e.g. [1]_.  You may also cite these
        references in the notes section above.

        .. [1] O. McNoleg, "The integration of GIS, remote sensing,
           expert systems and adaptive co-kriging for environmental habitat
           modelling of the Highland Haggis using object-oriented, fuzzy-logic
           and neural-network techniques," Computers & Geosciences, vol. 22,
           pp. 585-588, 1996.

        Examples
        --------
        These are written in doctest format, and should illustrate how to
        use the function.

        >>> a=[1,2,3]
        >>> print [x + 3 for x in a]
        [4, 5, 6]
        >>> print "a\n\nb"
        a
        b

        """
        return 0

    # Gets the output signal in the form (num_inputs, 1) and
    def get_output(self, module):
        return module.neurons.V.get() * self.non_map

class Parameter:
    """
    Interface representing each element of the composition of parameters.

    """
    self.__values = []

    def __len__(self):
        return self.__values.len()

    def __getitem__(self, index):
        return self.__values[index]

class ParametersComposition(Parameter):

    self.__elements = []

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

class Thredshold(Parameter):
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
