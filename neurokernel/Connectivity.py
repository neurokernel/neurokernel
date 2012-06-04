import numpy as np
import neurokernel.tools.misc_utils as mu

class Connectivity(object):
    """
    Module connectivity object.

    Describes connectivity between the public output neurons in a source
    module and the neurons in a destination module.

    Parameters
    ----------
    mat : numpy.ndarray of bools with shape (N, M)
        Boolean connectivity matrix describing connectivity between
        N source output neurons and M destination neurons.

    Notes
    -----
    The input connectivity matrix is assumed to be of the
    following form:
    
    in\out  0   1   2   3  
          +---+---+---+---+
       0  | x | x |   | x |
       1  |   |   | x |   |
       2  |   | x |   | x |        
       3  | x |   |   |   |  

    """
    
    def __init__(self, mat):
        
        if mat.dtype <> bool:
            raise IOError("Boolean connectivity matrix required.")
        if mat.ndim <> 2:
            raise IOError("2D connectivity matrix required.")
        self.mat = mat

        # Find the source neurons that have output connections:
        self.out_mask = np.any(mat, axis=1)

    @property
    def get_out_ind(self):
        """
        Return indices of source neurons with output connections.
        """

        return np.arange(self.mat.shape[0])[self.out_mask]

    @property
    def get_compressed(self):
        """
        Connectivity matrix with connectionless rows discarded.
        """

        # Discard the rows with no output connections:
        return np.compress(self.out_mask, self.mat, axis=0)
        
    def __repr__(self):
        return np.asarray(self.mat, int).__repr__()
    
def main():
    mat = mu.rand_bin_matrix((5, 10), 5, bool)
    return Connectivity(mat)

if __name__ == '__main__':
    c=main()
