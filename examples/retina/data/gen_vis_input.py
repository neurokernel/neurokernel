import h5py
import numpy as np

from eyeimpl import EyeGeomImpl

def generate_input(intensities_file, image_file, num_rings=17):
    """ generates inputs for photoreceptors
        intensities_file: the file where inputs will be written (h5 file)
        image_file: the file based on which inputs will be generated (mat file)
        num_rings: determines implicitly the number of photoreceptors
    """
    eye_geometry = EyeGeomImpl(num_rings)
    intensities = eye_geometry.get_intensities(image_file, 
                                               {'still_image': True})
    
    with h5py.File(intensities_file, 'w') as f:
        f.create_dataset('array', intensities.shape,
                         dtype=np.float64,
                         data=intensities)


if __name__ == '__main__':
    # use to generate custom input
    # currently generate_input is used externally
    pass