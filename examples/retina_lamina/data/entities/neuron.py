class Neuron(object):
    def __init__(self, params):
        """ params: a dictionary of neuron parameters.
                    It can be passed as an attribute dictionary parameter
                    for a node in networkx library.
        """
        self._params = params.copy()
        self._num = None
        
        # redundant parameter
        # shows the number of neurons to be defined is set
        if 'num' in self._params.keys():
            del self._params['num']

    @property
    def num(self):
        """ A positive number that should be unique among all neurons and 
            possibly used as a node identifier in networkx graph
        """
        return self._num
    
    @num.setter
    def num(self, value):
        try:
            if value < 0:
                self._num = None
            else:
                self._num = value
        except TypeError:
            self._num = None
            
    def is_dummy(self):
        return self._num is None

    @property
    def params(self):
        return self._params

    def update_selector(self, selector):
        # print("sel:{}".format(selector))
        self._params.update({'selector': selector})

    def add_param(self, key, val):
        self._params[key] = val
