from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from math import ceil
from scipy.io import loadmat

import h5py
import networkx as nx
import numpy as np
PI = np.pi

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FFMpegFileWriter
import matplotlib.pyplot as plt


import neurokernel.LPU.utils.simpleio as sio
from neurokernel.pattern import Pattern

from neurongeometry import NeuronGeometry
from image2signal import Image2Signal
from entities.neuron import Neuron
from entities.synapse import Synapse
from entities.hemisphere_ommatidium import (HemisphereOmmatidium, 
                                            HemisphereNeuron)
from models import model1

from map.mapimpl import (AlbersProjectionMap, EquidistantProjectionMap,
                         SphereToSphereMap)

from transform.imagetransform import ImageTransform

PORT_IN_GPOT = 'port_in_gpot'


class EyeGeomImpl(NeuronGeometry, Image2Signal):
    # used by static and non static methods and should
    # be consistent among all of them
    OMMATIDIUM_CLS = HemisphereOmmatidium
    DEFAULT_INTENSITY = 0
    MARGIN = 10

    def __init__(self, nrings, reye=1):
        """ map to sphere based on a 2D hex geometry that is closer to a circle
            e.g for one ring there will be 7 neurons arranged like this:
                4
             3     5
                0
             2     6
                1
            Every other ring will have +6 neurons

            nrings: number of rings
            r_eye: radius of eye hemisphere
                   (radius of screen is fixed to 10 right now)
        """
        self._nrings = nrings
        self._reye = reye

        self._nommatidia = self._get_globalid(nrings+1, 0)
        self._ommatidia = []
        self._supneighborslist = []
        self._adjneighborslist = []
        self._init_neurons()
        self._generate_retina()
        self._generate_lamina()



    @staticmethod
    def _get_globalid(ring, local_id):
        """ Gets a unique id of position based on the ring
            it belongs and its relative id in the ring
            e.g the first rings have local ids
            0 (ring 0)
            0 1 2 3 4 5 (ring 1)
            0 1 2 3 ...
            and global ids
            0
            1 2 3 4 5 6
            7 8 9 ...
        """
        if ring == 0:
            return 0
        else:
            return 3*(ring-1)*ring + 1 + (local_id % (6*ring))

    #TODO looks like the methods can be combined
    # define a method that returns id depending on position
    def _get_neighborgids_adjacency(self, lid, ring):
        """ Gets adjacent ids in the following order
                4
            3       5
                x
            2       6
                1
        """
        neighborgids = [0]*6

        # **** id1 ****
        id = self._get_neighborgid(lid, ring, 1)

        neighborgids[0] = id

        # **** id2 ****
        id = self._get_neighborgid(lid, ring, 2)

        neighborgids[1] = id

        # **** id3 ****
        id = self._get_neighborgid(lid, ring, 3)

        neighborgids[2] = id

        # **** id4 ****
        id = self._get_neighborgid(lid, ring, 5)

        neighborgids[3] = id

        # **** id5 ****
        id = self._get_neighborgid(lid, ring, 6)

        neighborgids[4] = id

        # **** id6 ****
        id = self._get_neighborgid(lid, ring, 7)

        neighborgids[5] = id

        return neighborgids

    def _get_neighborgids_superposition(self, lid, ring):
        """ Gets global ids of neighbors in the following order (see also 
            RFC2 figure 2)
            in neighbors
            (here x is cartridge and gets inputs from numbered neighbors)
            3
                4
            2       5
                x
            1       6
            ---------
            out neighbors
            (here x is ommatidium and sends output to numbered neighbors)
            6       1
                x
            5       2
                4
                    3
        """
        in_neighborgids = [0]*6
        out_neighborgids = [0]*6

        # **** in id1 out id5 ****
        id = self._get_neighborgid(lid, ring, 2)

        in_neighborgids[0] = id
        out_neighborgids[4] = id

        # **** in id2 out id6 ****
        id = self._get_neighborgid(lid, ring, 3)

        in_neighborgids[1] = id
        out_neighborgids[5] = id

        # **** in id3 ****
        id = self._get_neighborgid(lid, ring, 4)

        in_neighborgids[2] = id

        # **** out id3 ****
        id = self._get_neighborgid(lid, ring, 8)

        out_neighborgids[2] = id

        # **** in id4 ****
        id = self._get_neighborgid(lid, ring, 5)

        in_neighborgids[3] = id
        
        # **** out id4 ****
        id = self._get_neighborgid(lid, ring, 1)

        out_neighborgids[3] = id

        # **** in id5 out id1 ****
        id = self._get_neighborgid(lid, ring, 6)

        in_neighborgids[4] = id
        out_neighborgids[0] = id

        # **** in id6 out id2 ****
        id = self._get_neighborgid(lid, ring, 7)

        in_neighborgids[5] = id
        out_neighborgids[1] = id

        return (in_neighborgids, out_neighborgids)

    @classmethod
    def _get_neighborgid(cls, lid, ring, pos):
        """ Return the global id of one neighbbor at relative position pos
            pos: the relative position which can be from 0 to 8
            4
                5
            3       6
                0
            2       7
                1
                    8
        """
        # note lid is from 0 to 6*ring-1
        quot_ring, res_ring = divmod(lid, ring)

        if pos == 0:
            id = cls._get_globalid(ring, lid)
        elif pos == 1:
            if (quot_ring == 0) or \
                    ((quot_ring == 1) and (res_ring == 0)):
                id = cls._get_globalid(ring+1, lid)
            elif ((quot_ring == 1) and (res_ring > 0)) or \
                    ((quot_ring == 2) and (res_ring == 0)):
                id = cls._get_globalid(ring, lid-1)
            elif ((quot_ring == 2) and (res_ring > 0)) or \
                    (quot_ring == 3):
                id = cls._get_globalid(ring-1, lid-3)
            elif (quot_ring == 4):
                id = cls._get_globalid(ring, lid+1)
            elif (quot_ring == 5):
                id = cls._get_globalid(ring+1, lid+6)
        elif pos == 2:
            if (quot_ring == 0) or (quot_ring == 1) or \
                    ((quot_ring == 2) and (res_ring == 0)):
                id = cls._get_globalid(ring+1, lid+1)
            elif ((quot_ring == 2) and (res_ring > 0)) or \
                    ((quot_ring == 3) and (res_ring == 0)):
                id = cls._get_globalid(ring, lid-1)
            elif ((quot_ring == 3) and (res_ring > 0)) or \
                    (quot_ring == 4):
                id = cls._get_globalid(ring-1, lid-4)
            elif (quot_ring == 5):
                id = cls._get_globalid(ring, lid+1)
        elif pos == 3:
            if (quot_ring == 0):
                id = cls._get_globalid(ring, lid+1)
            elif (quot_ring == 1) or (quot_ring == 2) or \
                    ((quot_ring == 3) and (res_ring == 0)):
                id = cls._get_globalid(ring+1, lid+2)
            elif ((quot_ring == 3) and (res_ring > 0)) or \
                    ((quot_ring == 4) and (res_ring == 0)):
                id = cls._get_globalid(ring, lid-1)
            elif ((quot_ring == 4) and (res_ring > 0)) or \
                    (quot_ring == 5):
                id = cls._get_globalid(ring-1, lid-5)
        elif pos == 4:
            if ((quot_ring == 0) and (res_ring < ring-1)):
                id = cls._get_globalid(ring-1, lid+1)
            elif ((quot_ring == 0) and (res_ring == ring-1)):
                id = cls._get_globalid(ring, lid+2)
            elif (quot_ring == 1):
                id = cls._get_globalid(ring+1, lid+3)
            elif (quot_ring == 2) or ((quot_ring == 3) and (res_ring == 0)):
                id = cls._get_globalid(ring+2, lid+5)
            elif ((quot_ring == 3) and (res_ring > 0)) or \
                    ((quot_ring == 4) and (res_ring == 0)):
                id = cls._get_globalid(ring+1, lid+2)
            elif ((quot_ring == 4) and (res_ring == 1)) or \
                    ((quot_ring == 5) and (res_ring == 0) and (ring == 1)):
                id = cls._get_globalid(ring, lid-2)
            elif ((quot_ring == 4) and (res_ring > 1)) or \
                    ((quot_ring == 5) and (res_ring == 0)):
                id = cls._get_globalid(ring-1, lid-6)
            elif ((quot_ring == 5) and (res_ring > 0)):
                id = cls._get_globalid(ring-2, lid-11)
        elif pos == 5:
            if (quot_ring == 0):
                id = cls._get_globalid(ring-1, lid)
            elif (quot_ring == 1):
                id = cls._get_globalid(ring, lid+1)
            elif (quot_ring == 2) or (quot_ring == 3) or \
                    ((quot_ring == 4) and (res_ring == 0)):
                id = cls._get_globalid(ring+1, lid+3)
            elif ((quot_ring == 4) and (res_ring > 0)) or \
                    ((quot_ring == 5) and (res_ring == 0)):
                id = cls._get_globalid(ring, lid-1)
            elif ((quot_ring == 5) and (res_ring > 0)):
                id = cls._get_globalid(ring-1, lid-6)
        elif pos == 6:
            if ((quot_ring == 0) and (res_ring > 0)) or \
                    (quot_ring == 1):
                id = cls._get_globalid(ring-1, lid-1)
            elif (quot_ring == 2):
                id = cls._get_globalid(ring, lid+1)
            elif (quot_ring == 3) or (quot_ring == 4) or \
                    ((quot_ring == 5) and (res_ring == 0)):
                id = cls._get_globalid(ring+1, lid+4)
            elif ((quot_ring == 5) and (res_ring > 0)) or \
                    ((quot_ring == 0) and (res_ring == 0)):
                id = cls._get_globalid(ring, lid-1)
        elif pos == 7:
            if ((quot_ring == 0) and (res_ring == 0)):
                id = cls._get_globalid(ring+1, lid-1)
            elif ((quot_ring == 0) and (res_ring > 0)) or \
                    ((quot_ring == 1) and (res_ring == 0)):
                id = cls._get_globalid(ring, lid-1)
            elif ((quot_ring == 1) and (res_ring > 0)) or \
                    (quot_ring == 2):
                id = cls._get_globalid(ring-1, lid-2)
            elif (quot_ring == 3):
                id = cls._get_globalid(ring, lid+1)
            elif (quot_ring == 4) or (quot_ring == 5):
                id = cls._get_globalid(ring+1, lid+5)
        elif pos == 8:
            if ((quot_ring == 0) and (res_ring == 0)):
                id = cls._get_globalid(ring+2, lid-1)  # lid = 0 ->
                                                        # lid-1=-1 ->
                                                        # 6*(ring+2)-1
            elif ((quot_ring == 0) and (res_ring > 0)) or \
                    ((quot_ring == 1) and (res_ring == 0)):
                id = cls._get_globalid(ring+1, lid-1)
            elif ((quot_ring == 1) and (res_ring == 1)) or \
                    ((quot_ring == 2) and (res_ring == 0) and (ring == 1)):
                id = cls._get_globalid(ring, lid-2)
            elif ((quot_ring == 1) and (res_ring > 1)) or \
                    ((quot_ring == 2) and (res_ring == 0)):
                id = cls._get_globalid(ring-1, lid-3)
            elif ((quot_ring == 2) and (res_ring > 0)):
                id = cls._get_globalid(ring-2, lid-5)
            elif ((quot_ring == 3) and (res_ring < ring-1)):
                id = cls._get_globalid(ring-1, lid-2)
            elif ((quot_ring == 3) and (res_ring == ring-1)):
                id = cls._get_globalid(ring, lid+2)
            elif (quot_ring == 4):
                id = cls._get_globalid(ring+1, lid+6)
            elif (quot_ring == 5):
                id = cls._get_globalid(ring+2, lid+11)
        else:
            raise ValueError("Invalid position {}, should be 0-8")

        return id



    def _init_neurons(self):
        """ Generates neurons and initializes neighbors """
        nrings = self._nrings
        reye = self._reye
        ommatidium_cls = self.OMMATIDIUM_CLS
        ommatidia = self._ommatidia
        supneighborslist = self._supneighborslist
        adjneighborslist = self._adjneighborslist

        # the first neuron is a special case
        ommatid = 0
        lat = long = 0
        ommatidium = ommatidium_cls(lat, long, reye, ommatid)
        ommatidium.add_photoreceptor((lat, long), 0)

        ommatidia.append(ommatidium)
        supneighborslist.append([ommatidium])
        adjneighborslist.append([ommatidium])

        for ring in range(nrings + 2):
            # lid is local id
            # see _get_globalid method docstring
            ringP1 = ring + 1
            for lid in range(6*ringP1):
                self._update_ommatidia(ringP1, lid)
                self._update_neighbors(ringP1, lid)

    def _update_ommatidia(self, ring, lid):
        nrings = self._nrings
        reye = self._reye
        ommatidia = self._ommatidia
        ommatidium_cls = self.OMMATIDIUM_CLS

        lat = (ring/(nrings + 2))*(PI/2)  # lat: 0 to pi/2,
                                          # but we don't map near pi/2
        long = (lid/(6*ring))*2*PI - PI   # long: -pi to pi

        # if ommatidium is outside the range of rings,
        # construct it without id
        if ring > nrings:
            ommatidium = ommatidium_cls(lat, long, reye)
        else:
            gid = self._get_globalid(ring, lid)
            ommatidium = ommatidium_cls(lat, long, reye, gid)
            ommatidium.add_photoreceptor((lat, long), 0)

        ommatidia.append(ommatidium)

    def _update_neighbors(self, ring, lid):
        nrings = self._nrings
        nommatidia = self._nommatidia
        ommatidia = self._ommatidia
        supneighborslist = self._supneighborslist
        adjneighborslist = self._adjneighborslist

        gid = self._get_globalid(ring, lid)
        ommatidium = ommatidia[gid]

        # out neighbors (neighbors to send the output to
        #                from current ommatidium)
        # in neighbors (neighbors that respective cartridge receives input from)
        # supneighborslist has the out neighbors
        # for the relative position parameter, check the order that
        # the neighbors are returned by _get_neighborgids_superposition
        if ring > nrings:
            in_neighborgids, out_neighborgids = \
                self._get_neighborgids_superposition(lid, ring)
            adj_neighborgids = self._get_neighborgids_adjacency(lid, ring)
            for i, neighborgid in enumerate(in_neighborgids):
                if neighborgid < nommatidia:
                    neighborommat = ommatidia[neighborgid]
                    relative_neighborpos = i + 1
                    neighborommat.add_photoreceptor(ommatidium.get_direction(),
                                                       relative_neighborpos)
                    supneighborslist[neighborgid].append(ommatidium)
            for i, neighborgid in enumerate(adj_neighborgids):
                if neighborgid < nommatidia:
                    adjneighborslist[neighborgid].append(ommatidium)
        else:
            supneighbors = [ommatidium]
            adjneighbors = [ommatidium]
            in_neighborgids, out_neighborgids = \
                self._get_neighborgids_superposition(lid, ring)
            adj_neighborgids = self._get_neighborgids_adjacency(lid, ring)
            for i, neighborgid in enumerate(in_neighborgids):
                if neighborgid < gid:
                    neighborommat = ommatidia[neighborgid]
                    relative_neighborpos = i + 1
                    neighborommat.add_photoreceptor(ommatidium.get_direction(),
                                                    relative_neighborpos)
                    supneighborslist[neighborgid].append(ommatidium)
            for i, neighborgid in enumerate(out_neighborgids):
                if neighborgid < gid:
                    neighborommat = ommatidia[neighborgid]
                    relative_neighborpos = 5 - i
                    ommatidium.add_photoreceptor(neighborommat.get_direction(),
                                                 relative_neighborpos)
                    supneighbors.append(neighborommat)

            for i, neighborgid in enumerate(adj_neighborgids):
                if neighborgid < gid:
                    neighborommat = ommatidia[neighborgid]
                    adjneighborslist[neighborgid].append(ommatidium)
                    adjneighbors.append(neighborommat)

            supneighborslist.append(supneighbors)
            adjneighborslist.append(adjneighbors)

    def get_ommatidia(self):
        return self._ommatidia

    def get_neighbors(self, config = {'rule':'superposition'}):
        """
            config: configuration dictionary
                    keys
                    rule: "superposition"(default) neighbors are defined 
                          according to the superposition rule and correspond 
                          to the columns that receive input from the 
                          ommatidium (second image of Figure 2 in RFC2)
                          "adjacency" neighbors are the closest
            
            returns: a list of lists of neighbors, the entries in the initial
                     list are as many as the ommatidia and contain a list of
                     the neighboring ommatidia in no particular order except
                     the first entry that is the reference ommatidium
        """
        try:
            rule = config['rule']
            valid_values = ["superposition", "adjacency"]
            if coord not in valid_values:
                raise ValueError("rule attribute must be one of %s" % ", "
                                 .join(str(x) for x in valid_values))
        except KeyError:
            rule = "superposition"
        
        if rule == "superposition":
            return self._supneighborslist
        else:
            return self._adjneighborslist

    def get_positions(self, config={'coord': 'spherical', 'include': 'center',
                                    'add_dummy':True }):
        """ returns projected positions of photoreceptors
            config: configuration dictionary
                    keys
                    coord: "spherical"(default), returns
                           a tuple of 2 lists (latitudes, longitudes)
                           "cartesian3D", returns a tuple of 3 lists
                           (x, y, z)
                           "cartesian2D", returns a tuple of 2 lists
                           (x, y)
                    include: 'all', 'R1toR6' or 'center'
                    add_dummy: if True(default) 
                               adds 2 more levels of dummy neurons.
                               Useful for superposition rule.
            returns: tuple of lists depending on coordinate type.
                     Each entry in the lists corresponds to a photoreceptor.
                     The photoreceptors of the same ommatidium are consecutive.
                     The order of their positions is 
                     6       1
                         0
                     5       2
                         4
                             3
                     in case of 'center' only 0 is returned and in case of
                     'R1toR6' all others in numerical order
        """
        # the order of ommatidia in output is the exact order of
        # _ommatidia parameter
        try:
            coord = config['coord']
            valid_values = ["spherical", "cartesian3D", "cartesian2D"]
            if coord not in valid_values:
                raise ValueError("coord attribute must be one of %s" % ", "
                                 .join(str(x) for x in valid_values))
        except KeyError:
            coord = "spherical"

        try:
            include = config['include']
        except KeyError:
            include = 'center'
        
        try:
            add_dummy = config['add_dummy']
        except KeyError:
            add_dummy = True

        latitudes = []
        longitudes = []

        if include == 'all':
            for ommatidium in self._ommatidia:
                if add_dummy or not ommatidium.is_dummy():
                    screenlats, screenlongs = ommatidium.get_screenpoints()
                    latitudes.extend(screenlats)
                    longitudes.extend(screenlongs)
        elif include == 'R1toR6':
            for ommatidium in self._ommatidia:
                if add_dummy or not ommatidium.is_dummy():
                    screenlats, screenlongs = ommatidium.get_R1toR6points()
                    latitudes.extend(screenlats)
                    longitudes.extend(screenlongs)
        else:
            for ommatidium in self._ommatidia:
                if add_dummy or not ommatidium.is_dummy():
                    eyelat, eyelong = ommatidium.get_eyepoint()
                    latitudes.append(eyelat)
                    longitudes.append(eyelong)

        if coord == "spherical":
            return (np.array(latitudes), np.array(longitudes))
        elif coord == "cartesian3D":
            nplatitudes = np.array(latitudes)
            nplongitudes = np.array(longitudes)
            sinlats = np.sin(nplatitudes)
            coslats = np.cos(nplatitudes)
            sinlongs = np.sin(nplongitudes)
            coslongs = np.cos(nplongitudes)
            return (sinlats*coslongs, sinlats*sinlongs, coslats)
        elif coord == "cartesian2D":
            nplatitudes = np.array(latitudes)
            nplongitudes = np.array(longitudes)
            r2d = nplatitudes/(PI/2)
            sinlongs = np.sin(nplongitudes)
            coslongs = np.cos(nplongitudes)
            return (r2d*coslongs, r2d*sinlongs)

    def get_intensities(self, file, config={}):
        """ file: mat file is assumed but in case more file formats need to be
                  supported an additional field should be included in config
            config: {'still_image': True(default)/False, 
                     'steps':<simulation steps>,
                     'dt':<time step>, 'output_file':<filename>}
            returns: numpy array with with height the number of simulation
                     steps and width the number of neurons.
                     The order of neurons is the same as that returned 
                     in get_positions, with 'include'
                     configuration parameter set to 'R1toR6'
        """
        # the order of ommatidia in output is the exact order of
        # _ommatidia parameter
        try:
            dt = config['dt']
        except KeyError:
            dt = 1e-4

        try:
            time_steps = config['steps']
        except KeyError:
            time_steps = 1000

        try:
            still_image = config['still_image']
        except KeyError:
            still_image = True
            
        try:
            # TODO
            output_file = config['output_file']
        except KeyError:
            output_file = None

        mat = loadmat(file)
        try:
            image = np.array(mat['im'])
        except KeyError:
            print('No variable "im" in given mat file')
            print('Available variables (and meta-data): {}'.format(mat.keys()))

        # photons were stored for 1ms
        image *= (dt/1e-3)

        h_im, w_im = image.shape
        transform = ImageTransform(image)

        # screen positions
        # should be much more than the number of photoreceptors
        # shape (20, 3)*self._nrings
        screenlat, screenlong = np.meshgrid(np.linspace(0, PI/2,
                                                        3*self._nrings),
                                            np.linspace(-PI, PI,
                                                        20*self._nrings))

        mapscreen = self.OMMATIDIUM_CLS.MAP_SCREEN
        # plane positions
        # shape (20, 3)*self._nrings
        mx, my = mapscreen.map(screenlat, screenlong)   # map from spherical
                                                        # grid to plane

        # photoreceptor positions
        photorlat, photorlong = self.get_positions({'coord': 'spherical',
                                                    'include': 'R1toR6',
                                                    'add_dummy': False})

        if still_image:
            mx = mx - mx.min()  # start from 0
            my = my - my.min()
            mx *= h_im/mx.max()  # scale to image size
            my *= w_im/my.max()

            # shape (20, 3)*self._nrings
            transimage = transform.interpolate((mx, my))
            
            intensities = self._get_intensities(transimage, photorlat, 
                                                photorlong, screenlat, 
                                                screenlong)
            intensities = np.tile(intensities, (time_steps, 1))
        else:
            intensities = np.empty((time_steps, len(photorlat)), 
                                   dtype='float32')
            mx = mx - mx.min()  # start from 0
            my = my - my.min()
            for i in range(time_steps):
                mx = mx + 2*np.random.random() - 1  # move randomly
                my = my + 2*np.random.random() - 1  # between -1 and 1
                mxi_max = mx.max()
                if mxi_max > h_im:
                    mx = mx - 2*(mxi_max-h_im)
                mxi_min = mx.min()
                if mxi_min < 0:
                    mx = mx - 2*mxi_min

                myi_max = my.max()
                if myi_max > w_im:
                    my = my - 2*(myi_max-w_im)
                myi_min = my.min()
                if myi_min < 0:
                    my = my - 2*myi_min
                
                
                transimage = transform.interpolate((mx, my))
                intensities[i] = self._get_intensities(transimage, photorlat, 
                                                       photorlong, screenlat, 
                                                       screenlong)

        # get interpolated image values on these positions

        return intensities

    def _get_intensities(self, transimage, photorlat, photorlong,
                         screenlat, screenlong):
    
        intensities = np.empty(len(photorlat), dtype='float32')
        # shape (20, 3)*self._nrings
        # h meridians, w parallels
        h_tim, w_tim = transimage.shape
            
        for i, (lat, long) in enumerate(zip(photorlat, photorlong)):

            # position in grid (float values)
            parallelf = (w_tim - 1)*lat/(PI/2)  # float value
            meridianf = (h_tim - 1)*(long + PI)/(2*PI)  # float value

            # for each photoreceptor point find indexes of
            # nxn closest points on grid (int values)
            indlat, indlong = self.get_closest_indexes(parallelf, meridianf,
                                                       min1=0, min2=0, 
                                                       max1=h_tim, 
                                                       max2=w_tim, 
                                                       n=2)
            try:
                pixels = transimage[indlong, indlat]  # nxn np array
            except IndexError:
                print('Array size is {}'.format(transimage.shape))
                print('Indexes are {} and {}'.format(indlat, indlong))
                raise

            weights = self.get_gaussian_sphere(screenlat[indlong, indlat],
                                               screenlong[indlong, indlat],
                                               lat, long)
            # works with >=1.8 numpy
            intensities[i] = np.sum(pixels*weights)

        return intensities
            
            
    
    @staticmethod
    def get_closest_indexes(f1, f2, min1, min2, max1, max2, n):
        """ Given a point (f1, f2) return nxn closest points in the box
            [min1, min2, max1, max2]
        """
        ind1 = np.linspace(np.floor(f1) + (1-n/2), np.ceil(f1) + (-1+n/2), n) \
            .astype(int)
        ind2 = np.linspace(np.floor(f2) + (1-n/2), np.ceil(f2) + (-1+n/2), n) \
            .astype(int)
        ind1 = np.minimum(ind1, max1)
        ind2 = np.minimum(ind2, max2)

        ind1 = np.maximum(ind1, min1)
        ind2 = np.maximum(ind2, min2)
        return np.meshgrid(ind1, ind2)

    @staticmethod
    def get_gaussian_sphere(lats, longs, reflat, reflong):
        """ Computes gaussian function on sphere at points (lats, longs),
            with a given center (reflat, reflong).
            Values are normalized so that they sum up to 1.
            Kappa is 100
            
            see also:
            http://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
        """
        #
        KAPPA = 100
        # inner product of reference point with other points translated to
        # spherical coordinates
        in_prod = np.sin(lats)*np.sin(reflat)*np.cos(reflong-longs) \
            + np.cos(lats)*np.cos(reflat)
        func3 = np.exp(KAPPA*in_prod)
        # ignore constant factor C_p and normalize so that everything sums to 1
        return func3/np.sum(func3)

    #TODO apply DRY principle
    def visualise_output(self, model_output, media_file, config={}):
        """ config: { LPU: 'retina'(default)/'lamina'
                      type: 'image'(default)/'video'
                      neuron: the id of the neuron to plot(for lamina)
                              e.g L1
                    }
        """
        try:
            lpu = config['LPU']
        except KeyError:
            lpu = 'retina'
            
        try:
            type = config['type']
        except KeyError:
            type = 'image'
            
        with h5py.File(model_output, 'r') as f:
            data = f['array'].value
        
        if lpu == 'retina':
            xpositions, ypositions = self.get_positions(
                    {'coord': 'cartesian2D', 'include': 'R1toR6',
                     'add_dummy': False})

            self._plot_output(type, xpositions, ypositions, data, media_file)


        elif lpu == 'lamina':
            xpositions, ypositions = self.get_positions(
                    {'coord': 'cartesian2D', 'include': 'center',
                     'add_dummy': False})
            try:
                neuron = config['neuron']
            except KeyError:
                neuron = 'L1'

            n_ids = self._get_lamina_ids(neuron)

            print(n_ids)
            self._plot_output(type, xpositions, ypositions, data[:, n_ids],
                              media_file)
        else:
            raise ValueError('Invalid value for lpu: {}'
                             ', valid values retina, lamina'.format(lpu))

    def _plot_output(self, type, xpositions, ypositions, data, media_file):
        fig, ax = plt.subplots()

        if type == 'image':
            print(len(xpositions), len(data[-1]))
            print(data.shape)
            ax.scatter(xpositions, -ypositions, c=data[-1], cmap=cm.gray,
                       s=5, edgecolors='none')
                
            fig.savefig(media_file)
        elif type == 'video':
            writer = FFMpegFileWriter(fps=5, codec='libtheora')
            writer.setup(
                fig, media_file, dpi=80,
                frame_prefix=os.path.splitext(self.out_filename)[0]+'_')

            step = 100
            for i, d in enumerate(data):
                if i % step == 0:
                    ax.scatter(xpositions, -ypositions, c=data[i],
                               cmap=cm.gray, s=5, edgecolors='none')
                    fig.canvas.draw()
                    writer.grab_frame()
        else:
            raise ValueError('Invalid value for media type: {}'
                             ', valid values image, video'.format(type))

    def _get_lamina_ids(self, neuron_name):
        G = self._lamina_graph
        neurons = G.node.items()

        ids = []
        for id, neu in neurons:
            if neuron_name == neu['name']:
                ids.append(id)
        return ids

    def generate_input(self, image_file, output_file):
        intensities = self.get_intensities(image_file, 
                                           {'still_image': True})

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('array', intensities.shape,
                             dtype=np.float64,
                             data=intensities)
    
    def connect_retina_lamina(self, manager, ret_lpu, lam_lpu):
        #.values or .tolist()
        ret_sel = ret_lpu.interface.index.tolist()
        lam_sel = lam_lpu.interface.index.tolist()
        lam_sel_in = [sel for sel in lam_sel if sel[0]=="ret"]
        lam_sel_out = [sel for sel in lam_sel if sel[0]!="ret"]

        pat = Pattern(ret_sel, lam_sel)
        # pattern gets input from retina
        
        pat.interface[ret_sel, 'io'] = 'in'
        pat.interface[ret_sel, 'type'] = 'gpot'
        pat.interface[lam_sel_in, 'io'] = 'out'
        pat.interface[lam_sel_out, 'io'] = 'in'
        pat.interface[lam_sel, 'type'] = 'gpot'
        for sel in lam_sel_in:
            pat['/ret/out/' + str(sel[2]), '/ret/in/' + str(sel[2])] = 1
        
        # connect(self, m_0, m_1, pat, int_0=0, int_1=1):
        manager.connect(ret_lpu, lam_lpu, pat, 0, 1)

    def _generate_retina(self):
        G = nx.DiGraph()
        
        photoreceptor_num = 6*self._nommatidia

        for i in range(photoreceptor_num):
            G.node[i] = {
                'model': 'Photoreceptor',
                'name': 'photor' + str(i),
                'extern': True,  # gets input from file
                'public': True,  # it's an output neuron
                'spiking': False,
                'selector': '/ret/out/' + str(i),
                'num_microvilli': 30000
            }
        self._retina_graph = G

    #TODO break into smaller functions
    def _generate_lamina(self):
        G = nx.MultiDiGraph()

        # ommatidia are equal with the cartridges
        cartridge_num = self._nommatidia

        # lists will have only one copy of
        # neurons synapses
        # dicts may have duplicates
        neuron_list = []
        synapse_list = []
        neuron_dict = {}

        self._generate_neurons(neuron_list, neuron_dict, cartridge_num)

        supneighbors = self._supneighborslist

        # find the right selectors
        # supneighbors is list of lists
        for i, neighbors in enumerate(supneighbors):
            for j, neighbor in enumerate(neighbors[1:]):
                neuron_name = 'R' + str(j+1)
                # if neighbor is a valid ommatidium
                if not neighbor.is_dummy():
                    # photoreceptor j of ommatidium i sends output to
                    # respective neighbor, see rfc 2 figure 2
                    assert(neighbor.id < cartridge_num)
                    neuron = neuron_dict[(neighbor.id, neuron_name)]
                    photor_id = 6*i + j
                    # connected LPUs are not allowed to have the same name in
                    # ports in current implementation of pattern
                    neuron.update_selector('/ret/in/' + str(photor_id))
                    
        # renumber neurons to omit photoreceptors 
        # near the edge that have no input (no selector is set)
        count = 0
        for node in neuron_list:
            params = node.params
            if params['model'] == PORT_IN_GPOT:
                if 'selector' in params:
                    node.num = count
                    count += 1
                else:
                    node.num = None
            else:
                node.num = count
                count += 1

        self._generate_synapses(synapse_list, neuron_dict, cartridge_num)
        
        # add nodes to graph
        for node in neuron_list:
            # if neuron is not dummy
            if not node.is_dummy():
                G.add_node(node.num, node.params)

        for edge in synapse_list:
            edge.process_before_export()
            G.add_edge(edge.prenum, edge.postnum, attr_dict=edge.params)

        self._lamina_graph = G

    def _generate_neurons(self, n_list, n_dict, cartridge_num):
        neuron_types = model1.NEURON_LIST
        alpha_profiles = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

        for neuron_params in neuron_types:
            neuron_name = neuron_params['name']
            if neuron_name=='Am':
                am_num = neuron_params['num']
                bound = PI/np.sqrt(am_num)

                # create amacrine neurons
                for i in range(am_num):
                    n = Neuron(neuron_params)
                    n.num = len(n_list)
                    n_list.append(n)
                    n_dict[(i, 'Am')] = n

                # connect them to cartridges
                am_lats = (PI/2)*np.random.random(am_num)
                am_longs = 2*PI*np.random.random(am_num) - PI
                om_lats, om_longs = self.get_positions({'add_dummy': False})
                
                for i, (om_lat, om_long) in \
                        enumerate(zip(om_lats.tolist(), om_longs.tolist())):
                    # calculate distance and find amacrine cells within
                    # distance defined by bound
                    dist = np.sqrt((om_lat-am_lats)**2 + (om_long-am_longs)**2)
                    suitable_am = np.nonzero(dist <= bound)[0]
                    # if less than 4 neurons in the bound, get
                    # the 4 closest amacrine cells
                    if suitable_am.size < 4:
                        suitable_am = np.argsort(dist)[0:4]

                    # an amacrine neuron should not be connected more than 3
                    # times with a cartridge
                    fill = np.zeros(am_num, np.int8)
                    for name in alpha_profiles:
                        assigned = False
                        for am_ind in np.random.permutation(suitable_am):
                            if fill[am_ind] < 3:
                                fill[am_ind] += 1
                                n_dict[(i, name)] = \
                                    n_dict[(am_ind, 'Am')]
                                assigned = True
                                break
                        if not assigned:
                            print ("{} in cartridge {} not assigned"
                                   .format(name, i))
            else:
                # the order of neurons of the same name
                # is the same as the order of cartridges
                for i in range(cartridge_num):
                    n = Neuron(neuron_params)
                    n.num = len(n_list)
                    if neuron_params['output']:
                        n.update_selector('/lam/' + neuron_name + '/' + str(i))
                    n_list.append(n)
                    n_dict[(i, neuron_name)] = n

    def _generate_synapses(self, s_list, n_dict, cartridge_num):
        synapse_types = model1.SYNAPSE_LIST
        adjneighbors = self._adjneighborslist

        for synapse_params in synapse_types:
            prename = synapse_params['prename']
            postname = synapse_params['postname']
            cart = synapse_params['cart']

            if cart is None:
                for cart_num in range(cartridge_num):
                    s = Synapse(synapse_params)
                    
                    npre = n_dict[(cart_num, prename)]
                    npost = n_dict[(cart_num, postname)]
                    if npre.is_dummy() or npost.is_dummy():
                        continue
                    
                    s.prenum = npre.num
                    s.postnum = npost.num
                    s.update_class(Synapse.get_class(npre, npost))
                    s_list.append(s)
            else:
                for cart_num in range(cartridge_num):
                    s = Synapse(synapse_params)
                    
                    npre = n_dict[(cart_num, prename)]
                    if npre.is_dummy():
                        continue
                    s.prenum = npre.num
                    # find the neighbor cartridge
                    neighbor = adjneighbors[cart_num][cart]
                    # if cartridge is not dummy
                    if not neighbor.is_dummy():
                        npost = n_dict[(neighbor.id, postname)]
                        if npost.is_dummy():
                            continue
                        s.postnum = npost.num
                        s.update_class(Synapse.get_class(npre, npost))
                        s_list.append(s)

    def write_retina(self, output_file):
        nx.write_gexf(self._retina_graph, output_file)

    def write_lamina(self, output_file):
        nx.write_gexf(self._lamina_graph, output_file, prettyprint=True)


if __name__ == '__main__':
    from matplotlib.collections import LineCollection
    import time

    N_RINGS = 1
    STILL_IMAGE = True
    
    print('Initializing geometry')
    hemisphere = EyeGeomImpl(N_RINGS)
    print('Getting positions')
    positions = hemisphere.get_positions({'coord': "cartesian3D"})
    print('Getting neighbors')
    neighbors = hemisphere.get_neighbors()

    print('Getting intensities')
    start_time = time.time()
    intensities = hemisphere.get_intensities('image1.mat', 
                                             {'still_image': STILL_IMAGE})
    print("--- Duration {} seconds ---".format(time.time() - start_time))

    print('Setting up plot 1: Ommatidia on 3D sphere')
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    ax1.set_title('Ommatidia on 3D sphere')
    ax1.plot_trisurf(positions[0], positions[1], positions[2],
                     cmap=cm.coolwarm, linewidth=0.2)

    #

    print('Setting up plot 2: a) 2d projection of ommatidia and their'
          ' connections with neighbors, b) projections of screen positions')
    fig2 = plt.figure()
    ax2_1 = fig2.add_subplot(121)
    ax2_1.set_title('2d projection of ommatidia and their'
                    ' connections with neighbors')

    xpositions, ypositions = hemisphere.get_positions({'coord': "cartesian2D"})

    segmentlist = []
    for neighborlist in neighbors:
        original = neighborlist[0]
        original = original.id
        for neighbor in neighborlist[1:]:
            neighbor = neighbor.id
            if (neighbor != -1):  # some edges are added twice
                segmentlist.append([[xpositions[original],
                                     ypositions[original]],
                                    [xpositions[neighbor],
                                     ypositions[neighbor]]])

    linesegments = LineCollection(segmentlist, linestyles='solid')
    ax2_1.add_collection(linesegments)
    ax2_1.set_xlim((-1, 1))
    ax2_1.set_ylim((-1, 1))

    ax2_2 = fig2.add_subplot(122)
    ax2_2.set_title('2d projection of ommatidia and their'
                    ' projections of screen positions')
    xpositions, ypositions = hemisphere.get_positions({'coord': "cartesian2D",
                                                       'include': 'all',
                                                       'add_dummy': False})

    ax2_2.scatter(xpositions, ypositions)

    print('Setting up plot 3: greyscale image and intensities')

    mat = loadmat('image1.mat')
    try:
        image = np.array(mat['im'])
    except KeyError:
        print('No variable "im" in given mat file')
        print('Available variables (and meta-data): {}'.format(mat.keys()))

    fig3 = plt.figure()
    ax3_1 = fig3.add_subplot(121)
    ax3_1.set_title('greyscale image')
    ax3_1.imshow(image, cmap=cm.Greys_r)

    print(intensities.shape)
    print(intensities)
    ax3_2 = fig3.add_subplot(122)
    ax3_2.set_title('intensities')
    xpositions, ypositions = hemisphere.get_positions({'coord': "cartesian2D",
                                                       'include': 'R1toR6',
                                                       'add_dummy': False})
    print(xpositions.shape)

    ax3_2.scatter(xpositions, -ypositions, c=intensities[1], cmap=cm.gray, s=5,
                  edgecolors='none')

    plt.show()
