from abc import ABCMeta, abstractmethod


class PointMap(object):
    """ Interface of mapping a point from one surface to another
        (hence the 2 parameters) TODO make more generic
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, p1, p2):
        """ map point (p1,p2) from one surface to another """
        return

    @abstractmethod
    def invmap(self, p1, p2):
        """ inverse map point (p1,p2) from one surface to another """
        return
