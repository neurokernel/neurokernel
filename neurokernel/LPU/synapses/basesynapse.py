from abc import ABCMeta, abstractmethod, abstractproperty

class BaseSynapse(object):
    __metaclass__ = ABCMeta

    def __init__(self, s_dict, synapse_state_pointer, dt, debug):
        '''

        '''

    @abstractmethod
    def update_state(self, buffer):
        '''
        '''
        pass


    @abstractproperty
    def synapse_class(self):
        pass


    def post_run(self):
        pass