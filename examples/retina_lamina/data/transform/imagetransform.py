from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np

from scipy.ndimage.interpolation import map_coordinates
# from scipy.interpolate import interp2d

from signaltransform import SignalTransform


class ImageTransform(SignalTransform):
    def __init__(self, image):
        # tried to use interp2d but the interpolated coordinates
        # are translated to grid if they are one dimensional
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html
        self._image = image

    def get_transform(self):
        return self._image

    def interpolate(self, points):
        points0 = np.atleast_1d(points[0])
        points1 = np.atleast_1d(points[1])
        # TODO slow
        return map_coordinates(self._image, [points0, points1], order=1, 
                               prefilter=False)

if __name__ == '__main__':
    from scipy.io import loadmat
    import matplotlib.pyplot as plt

    mat = loadmat('../image1.mat')

    try:
        image = np.array(mat['im'])
        print('Image was loaded successfully, size: {}'.format(image.shape))
    except KeyError:
        print('No variable "im" in given mat file')
        print('Available variables (and meta-data): {}'.format(mat.keys()))

    print('Image values in rectangle from point {} to {}'
          .format((0, 0), (4, 4)))
    print('----------------------------')
    print(image[0:5, 0:5])
    print('----------------------------')

    trans = ImageTransform(image)
    point = (2.2, 2.3)
    pointvalue = trans.interpolate(point)
    print('Point {} has interpolated value {}'.format(point, pointvalue))

    # pointsx = np.array([1.5, 2.5])
    # pointsy = np.array([1.5, 2.5])
    pointsx = 10*np.random.random(16000).reshape(40,400)
    pointsy = 10*np.random.random(16000).reshape(40,400)
    points = (pointsx, pointsy)
    for _ in range(4000):
        pointsvalues = trans.interpolate(points)

    # print('Points {} have interpolated values {}'.format(points, pointsvalues))
