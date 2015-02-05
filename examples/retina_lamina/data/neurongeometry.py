from abc import ABCMeta, abstractmethod


class NeuronGeometry(object):
    """ Interface expected to be used by classes that
        define a geometry of photoreceptors
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_neighbors(self):
        """ Gets the ids of the connections of each neuron """
        return

    @abstractmethod
    def get_positions(self, config=None):
        """ Gets the position of each neuron.
            config: optional configuration
            (but class should have a default one), may be a custom
            object that may specify among others: the type of coordinates,
            whether include boundary corners(if applicable)
        """
        return
