class Synapse(object):
    @staticmethod
    def get_class(preneuron, postneuron):
        """ preneuron: Neuron instance 
            postneuron: Neuron instance 
        """
        is_pre_spk = preneuron.params['spiking']
        is_post_spk = postneuron.params['spiking']
        
        if is_pre_spk and is_post_spk:
            return 0
        elif is_pre_spk and not is_post_spk:
            return 1
        elif not is_pre_spk and is_post_spk:
            return 2
        elif not is_pre_spk and not is_post_spk:
            return 3
    def __init__(self, params):
        """ params: a dictionary of neuron parameters.
                    It can be passed as an attribute dictionary parameter
                    for a node in networkx library.
        """
        self._params = params.copy()

    @property
    def prenum(self):
        return self._prenum
    
    @prenum.setter
    def prenum(self, value):
        self._prenum = value

    @property
    def postnum(self):
        return self._postnum
    
    @postnum.setter
    def postnum(self, value):
        self._postnum = value

    @property
    def params(self):
        return self._params
    
    def update_class(self, cls):
        self._params.update({'class': cls})
        
    def process_before_export(self):
        self._params.update({'conductance': True})
        if 'cart' in self._params.keys():
            del self._params['cart']
        if 'scale' in self.params.keys():
            self._params['slope'] *= self._params['scale']
            self._params['saturation'] *= self._params['scale']
            del self._params['scale']