from abc import ABCMeta, abstractmethod


class Image2Signal(object):
    """ Interface expected to be used by classes that
        define a geometry of photoreceptors or any class
        that can define a map from a visual scene to photon intensity
        and backwards
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_intensities(self, file, config=None):
        """ given an image or other type of file
            get the photon intensities of neurons

            file:   input file
            config: optional configuration
                    (but class should have a default one), may be a custom
                    object that may specify among others: the type of file,
                    whether the output is of a moving image or not,
                    size of output. (Note part of the configuration may be
                    defined in the constructor)
        """
        return

    @abstractmethod
    def visualise_output(self, model_output, media_file, config=None):
        """ given the output of a neuron model generate a file
            that visualizes it

            model_output: output of neuron model
            media_file: output file (where visualized output will be stored)
            config: optional configuration
                    (but class should have a default one), may be a custom
                    object that may specify among others: the type of file,
                    whether the visualization is of a moving image or not,
                    type of output(spiking, non-spiking etc).
                    (Note part of the configuration may be
                    defined in the constructor)
        """
        return
