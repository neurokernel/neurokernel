from abc import ABCMeta, abstractmethod


class SignalTransform(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_transform(self):
        """ anything that represents the transformed signal
            coefficients etc.
        """
        return

    @abstractmethod
    def interpolate(self, point):
        """ gets the value of the signal at the specific point,
            as the signal is discrete it will be a kind of approximation
        """
        return
