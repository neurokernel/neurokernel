import numpy as np

class Connectivity:

    def __init__(self, mapping, proj_module, input_module):

        # 2D array (bool) as shown below:
        #       out1  out2  out3  out4
        # in1 |  x  |  x  |     |  x
        # in2 |     |     |  x  |
        # in3 |     |  x  |     |  x
        if mapping.dtype <> bool:
            raise IOError, "You must provide a mapping as a numpy.darray with \
                            booleans"
        if mapping.ndim <> 2:
            raise IOError, "You must provide a 2D numpy.darray"
        self.map = mapping

        # support vector in the form:
        # | True | False | False | True | True
        # where True or False mean if it is necessary to transmit or not the
        # state of determined neuron
        map_support = self.map.sum(axis = 0).astype(np.bool)
        # This mask contains the indices of neurons that will have their states
        # transmitted to the other LPU and will be stored at the same GPU of
        # the sender module.
        self.output_mask = np.compress(map_support, map_support * \
                                       range(1, len(map_support) + 1)) - 1
        # The compressed matrices will be stored at the same GPU of the
        # receiver module and represent the connectivity between the neurons
        # that are indeed connected. E.g.: if one module has 100 projection
        # neurons, but just 40 neurons are connected to a module, the matrices
        # need to have a shape like (num_receivers, 40) to map the connections.
        self.compressed_map = np.compress(map_support, self.map, axis = 1)

#        self.syn_parameters

    def add_connectivity(self, connectivity):
        self.connectivities.append(connectivity)

    def rm_connectivity(self, connecivity):
        self.connectivities.remove(connecivity)

    # Gets the output signal in the form (num_inputs, 1) and
    def get_output(self, module):
        return module.neurons.V.get() * self.map

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
