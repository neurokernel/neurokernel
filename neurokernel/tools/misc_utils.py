import numpy as np

def rand_bin_matrix(sh, N, dtype=np.double):
    """
    Generate a rectangular binary matrix with randomly distributed nonzero entries.

    Parameters
    ----------
    sh : tuple
        Shape of generated matrix.
    N : int
        Number of entries to set to 1.
    dtype : dtype
        Generated matrix data type.

    """

    result = np.zeros(sh, dtype)
    indices = np.arange(result.size)
    np.random.shuffle(indices)
    result.ravel()[indices[:N]] = 1
    return result
