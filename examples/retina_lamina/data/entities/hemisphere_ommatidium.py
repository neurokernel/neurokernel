from ..map.mapimpl import AlbersProjectionMap, SphereToSphereMap

class HemisphereOmmatidium(object):
    # static variables (Don't modify)
    _rscreen = 10
    # allows easy change to another map
    MAP_SCREEN = AlbersProjectionMap(_rscreen)

    def __init__(self, lat, long, reye, id=None):
        """ id: defaults to None, 
                which indicates that the ommatidium is dummy
        """
        self._id = id
        self._lat = lat
        self._long = long

        self._reye = reye
        self._neurons = []
        self._screenlats = [0]*7
        self._screenlongs = [0]*7

    @property
    def id(self):
        return self._id 

    def is_dummy(self):
        return self._id is None

    def add_photoreceptor(self, direction, rel_pos):
        """ direction: the direction photoreceptor will be facing
            rel_pos: the id of relative position of photoreceptor
            6       1
                0
            5       2
                4
                    3
            [the direction of photoreceptor 1 should be the direction 
             that neighbor at position 5 faces and the same applies 
             to all the others]
        """
        reye = self._reye
        rscreen = self._rscreen

        # eye to screen
        mapeye = SphereToSphereMap(reye, rscreen, direction)
        # screen to xy plane
        mapscreen = self.MAP_SCREEN

        neuron = HemisphereNeuron(mapeye, mapscreen, self._lat, self._long)
        self._neurons.append(neuron)

        # update screenpoints
        neuron_point = neuron.get_screenpoint()
        self._screenlats[rel_pos] = neuron_point[0]
        self._screenlongs[rel_pos] = neuron_point[1]

    def get_direction(self):
        return (self._lat, self._long)

    def get_eyepoint(self):
        return (self._lat, self._long)

    def get_screenpoints(self):
        return (self._screenlats, self._screenlongs)
        
    def get_R1toR6points(self):
        return (self._screenlats[1:], self._screenlongs[1:])


class HemisphereNeuron(object):
    """
        stores and transforms position related information
        access and set parameters only through the provided functions
    """
    def __init__(self, eye_to_screen_map, screen_to_2d_map,
                 eyelat, eyelong):

        self._eyelat = eyelat
        self._eyelong = eyelong
        screenlat, screenlong = eye_to_screen_map.map(eyelat, eyelong)
        self._screenlat = screenlat
        self._screenlong = screenlong
        x, y = screen_to_2d_map.map(screenlat, screenlong)
        self._x = x
        self._y = y

    def get_planepoint(self):
        return (self._x, self._y)

    def get_screenpoint(self):
        return (self._screenlat, self._screenlong)

    def append_screenpoint(self, screenlats, screenlongs):
        """ appends neuron's screen position to the respective parameters """
        screenlats.append(self._screenlat)
        screenlongs.append(self._screenlong)
