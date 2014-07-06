from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
#from math import pi, cos, sin, acos, atan2, sqrt

from pointmap import PointMap


class AlbersProjectionMap(PointMap):
    """ https://en.wikipedia.org/wiki/Albers_projection
    """
    def __init__(self, r=1):
        """ r: radious of the sphere from which the projection is made
        """
        self._r = r

    def map(self, lat, long):
        """ Returns (nan, nan) if point cannot be mapped else (x, y)
            arguments can be numpy arrays or scalars
        """
        rxy = self._r*np.sqrt(1-np.cos(lat))
        x = rxy*np.cos(long)
        y = rxy*np.sin(long)
        return (x, y)

    def invmap(self, x, y):
        """ Returns (nan, nan) if point cannot be mapped else
            (latitude, longitude)
            arguments can be numpy arrays or scalars
        """
        r = self._r
        rxy = np.sqrt(x*x + y*y)

        lat = np.arccos(1-(rxy/r)**2)
        long = np.arctan2(y, x)

        try:
            long[np.isnan(lat)] = np.nan
        except TypeError:  # Thrown if long is scalar
            if np.isnan(lat): long = np.nan
        return (lat, long)


class EquidistantProjectionMap(PointMap):
    """ https://en.wikipedia.org/wiki/Equidistant_conic_projection
    """
    def __init__(self, r=1):
        """ r: radious of the sphere from which the projection is made
        """
        self._r = r

    def map(self, lat, long):
        """ Returns (nan, nan) if point cannot be mapped else (x, y)
            arguments can be numpy arrays or scalars
        """
        rxy = self._r*lat/(np.pi/2)
        x = rxy*np.cos(long)
        y = rxy*np.sin(long)
        return (x, y)

    def invmap(self, x, y):
        """ Returns (nan, nan) if point cannot be mapped else
            (latitude, longitude)
            arguments can be numpy arrays or scalars
        """
        r = self._r
        rxy = np.sqrt(x*x + y*y)

        lat = (rxy/r)*(np.pi/2)
        long = np.arctan2(y, x)

        return (lat, long)


class SphereToSphereMap(PointMap):
    """ Map points from one screen to another.
        Screens are cocentered hemispheres
    """
    def __init__(self, r1, r2, direction):
        """ r1: initial screen radius
            r2: projected screen radius
            direction: can be a tuple or a list of 2 elements
                       latitude, longitude
        """
        self._r1 = r1
        self._r2 = r2
        self._direction = direction

    @staticmethod
    def map_aux(lat, long, direction, r1, r2):
        """ In case of 2 points of intersection picks the closest
        """

        dlat, dlong = direction

        # Transform from spherical to Cartesian coordinates
        # lat = 0 on z axis
        x1 = r1*np.sin(lat)*np.cos(long)
        y1 = r1*np.sin(lat)*np.sin(long)
        z1 = r1*np.cos(lat)

        dx1 = np.sin(dlat)*np.cos(dlong)
        dy1 = np.sin(dlat)*np.sin(dlong)
        dz1 = np.cos(dlat)
        
        res_shape = x1.shape

        xdx = x1*dx1 + y1*dy1 + z1*dz1

        # D may be <0 in this case nan will be returned
        D = xdx*xdx - r1*r1 + r2*r2
        # select the minimum of the 2 roots
        l12 = np.array([-xdx + np.sqrt(D), -xdx - np.sqrt(D)]).reshape(2, -1)
        indxl = np.argmin(abs(l12), axis=0)
        indyl = np.indices(indxl.shape)[0]

        # l is how far to move in direction dx
        l = l12[indxl, indyl].reshape(res_shape)

        x2 = x1 + l*dx1
        y2 = y1 + l*dy1
        z2 = z1 + l*dz1

        map_lat = np.arccos(z2/r2)
        map_long = np.arctan2(y2, x2)

        try:
            return (map_lat.reshape(res_shape), map_long.reshape(res_shape))
        except AttributeError:  # if lat and long scalars
            return (np.asscalar(map_lat), np.asscalar(map_long))

    def map(self, lat, long):
        """ Returns closest point of intersection or nan if there is none
            lat: the latitude on sphere1, can be scalar or numpy array
            long: the longitude on sphere1, can be scalar or numpy array

            returns: tuple of arrays of the same size as the initial ones
                    (that means even if input is scalar)
        """
        r1 = self._r1
        r2 = self._r2
        direction = self._direction

        return self.map_aux(lat, long, direction, r1, r2)

    def invmap(self, lat, long):
        """ Similar to map but map happens from sphere 2 to 1
        """
        r1 = self._r1
        r2 = self._r2
        direction = self._direction

        return self.map_aux(lat, long, direction, r2, r1)


if __name__ == '__main__':
    def test_map1(map, lat, long):
        print('-----------------------')
        print('Testing map class: {}'.format(type(map).__name__))

        print('-----------------------')
        print('Mapping (latitude:{} pi, longitude:{} pi) to plane'
              .format(lat/np.pi, long/np.pi))
        x, y = map.map(lat, long)
        print('Map result: (x:{}, y:{})'.format(x, y))

        print('-----------------------')
        print('Mapping (x:{}, y:{}) to sphere'.format(x, y))
        lat, long = map.invmap(x, y)
        print('Map result: (latitude:{} pi, longitude:{} pi)'
              .format(lat/np.pi, long/np.pi))

    def test_map2(map, lat, long):
        print('-----------------------')
        print('Testing map class: {}'.format(type(map).__name__))
        print(('Mapping (latitude:{} pi, longitude:{} pi)'
               ' from first hemisphere to second')
              .format(lat/np.pi, long/np.pi))
        lat, long = map.map(lat, long)
        print('Map result: (latitude:{} pi, longitude:{} pi)'
              .format(lat/np.pi, long/np.pi))

        print('-----------------------')
        print(('Mapping (latitude:{} pi, longitude:{} pi)'
               ' from second hemisphere to first')
              .format(lat/np.pi, long/np.pi, r2))
        lat, long = map.invmap(lat, long)
        print('Map result: (latitude:{} pi, longitude:{} pi)'
              .format(lat/np.pi, long/np.pi))

    test_data = [(0, 0), (0.2, 0.4), (np.array([0, 0.2]), np.array([0, 0.4]))]
    r = 2
    r1 = 1
    r2 = 10
    direction = (0, 0)

    for lat, long in test_data:
        print('=======================')
        print('Testing maps from hemisphere of radious {} to plane'.format(r))

        albermap = AlbersProjectionMap(r)
        test_map1(albermap, lat, long)

        equidistmap = EquidistantProjectionMap(r)
        test_map1(equidistmap, lat, long)

        #-----

        print('=======================')

        print(('Testing map from hemisphere of radious {} to hemisphere '
               'of radious {} in direction {}').format(r1, r2, direction))

        spheremap = SphereToSphereMap(r1, r2, direction)
        test_map2(spheremap, lat, long)
