#from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os

import itertools

from collections import namedtuple
from scipy.io import loadmat

import h5py
import networkx as nx
import math
import numpy as np
PI = np.pi

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FFMpegFileWriter, AVConvFileWriter
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize

import neurokernel.LPU.utils.simpleio as sio
from neurokernel.pattern import Pattern

from neurongeometry import NeuronGeometry
from image2signal import Image2Signal
from entities.neuron import Neuron
from entities.synapse import Synapse
from entities.hemisphere_ommatidium import (HemisphereOmmatidium,
                                            HemisphereNeuron)
from models import model_lamina, model_medulla

from map.mapimpl import (AlbersProjectionMap, EquidistantProjectionMap,
                         SphereToSphereMap)

from transform.imagetransform import ImageTransform

PORT_IN_GPOT = 'port_in_gpot'

import time
import pickle

class EyeGeomImpl(NeuronGeometry):
    # used by static and non static methods and should
    # be consistent among all of them
    OMMATIDIUM_CLS = HemisphereOmmatidium

    PARALLELS = 50
    MERIDIANS = 800

    INTENSITIES_FILE = 'intensities.h5'
    POSITIONS_FILE = 'positions.h5'
    IMAGE_FILE = 'image.h5'

    # TODO pass the names as parameters in case more configurations need to be stored
    RET_LAM_PAT_FILE = 'ret_lam_pat.pkl'
    LAM_MED_PAT_FILE = 'lam_med_pat.pkl'

    LPU_ORDER = {'r':0, 'l':1, 'm':2}
    
    #changed __init__, _get_neighborgids_adjacency,
    #_get_neighborgid, description and
    # _get_neighborgids_superposition source x->(x+3)%8 + 1
    def __init__(self, nrings, reye=1, model='r'):
        """ map to sphere based on a 2D hex geometry that is closer to a circle
            e.g for one ring there will be 7 neurons arranged like this:
                1
             6     2
                0
             5     3
                4
            Every other ring will have +6 neurons

            nrings: number of rings
            r_eye: radius of eye hemisphere
                   (radius of screen is fixed to 10 right now)
            model: should be a string with characters corresponding to the
                   LPUs to be simulated: r(etina), l(amina), m(edulla)
        """
        self._nrings = nrings
        self._reye = reye

        self._nommatidia = self._get_globalid(nrings+1, 0)
        self._ommatidia = []
        self._supneighborslist = []
        self._adjneighborslist = []
        self._init_neurons()
        
        # find first and last lpu according to the order they are connected
        first_lpu = LPU_ORDER['m']
        last_lpu = LPU_ORDER['r']
        for c in model:
            ord_c = LPU_ORDER[c]
            if ord_c < first_lpu:
                first_lpu = ord_c
            if ord_c > last_lpu:
                last_lpu = ord_c

        assert(last_lpu-first_lpu+1 == len(model))
        # numbers not characters    
        self._first_lpu = first_lpu
        self._last_lpu = last_lpu

        if first_lpu <= LPU_ORDER['r'] and last_lpu >= LPU_ORDER['r']:
            self._generate_retina()
        if first_lpu <= LPU_ORDER['l'] and last_lpu >= LPU_ORDER['l']:
            self._generate_lamina()
        if first_lpu <= LPU_ORDER['m'] and last_lpu >= LPU_ORDER['m']:
            self._generate_medulla()
            
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

    def _get_neighborgids_adjacency(self, lid, ring):
        """ Gets adjacent ids in the following order
                1
            6       2
                x
            5       3
                4
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
        id = self._get_neighborgid(lid, ring, 6)

        in_neighborgids[0] = id
        out_neighborgids[4] = id

        # **** in id2 out id6 ****
        id = self._get_neighborgid(lid, ring, 7)

        in_neighborgids[1] = id
        out_neighborgids[5] = id

        # **** in id3 ****
        id = self._get_neighborgid(lid, ring, 8)

        in_neighborgids[2] = id

        # **** out id3 ****
        id = self._get_neighborgid(lid, ring, 4)

        out_neighborgids[2] = id

        # **** in id4 ****
        id = self._get_neighborgid(lid, ring, 1)

        in_neighborgids[3] = id

        # **** out id4 ****
        id = self._get_neighborgid(lid, ring, 5)

        out_neighborgids[3] = id

        # **** in id5 out id1 ****
        id = self._get_neighborgid(lid, ring, 2)

        in_neighborgids[4] = id
        out_neighborgids[0] = id

        # **** in id6 out id2 ****
        id = self._get_neighborgid(lid, ring, 3)

        in_neighborgids[5] = id
        out_neighborgids[1] = id

        return (in_neighborgids, out_neighborgids)

    @classmethod
    def _get_neighborgid(cls, lid, ring, pos):
        """ Return the global id of one neighbbor at relative position pos
            pos: the relative position which can be from 0 to 8
            8
                1
            7       2
                0
            6       3
                5
                    4
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
                                                       # lid-1 = -1 ->
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
            raise ValueError("Invalid position {}, should be 0-8".format(pos))

        return id

    def _init_neurons(self):
        """ Generates neurons and initializes neighbors """
        nrings = self._nrings
        reye = self._reye
        ommatidium_cls = self.OMMATIDIUM_CLS
        ommatidia = self._ommatidia
        supneighborslist = self._supneighborslist
        adjneighborslist = self._adjneighborslist
        self._cart_coord = {}
        self._spher_coord = {}
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

        '''
        lat = (ring/(nrings + 2))*(PI/2)  # lat: 0 to pi/2,
                                          # but we don't map near pi/2
        long = (lid/(6*ring))*2*PI - PI   # long: -pi to pi
        
        '''
        x, y = self._get_cartesian_omm_loc(ring, lid)
        lat, long =  AlbersProjectionMap(self._reye).invmap(x, y)
        
        # if ommatidium is outside the range of rings,
        # construct it without id
        if ring > nrings:
            ommatidium = ommatidium_cls(lat, long, reye)
        else:
            gid = self._get_globalid(ring, lid)
            ommatidium = ommatidium_cls(lat, long, reye, gid)
            ommatidium.add_photoreceptor((lat, long), 0)
            self._cart_coord[gid] = (lat, long)
            self._spher_coord[gid] = (lat, long)
            
        ommatidia.append(ommatidium)

    def _get_cartesian_omm_loc(self, ring, lid):
        r = self._reye/np.sqrt(3)/self._nrings
        d1 = np.asarray([0, np.sqrt(3)*r])
        d2 = np.asarray([1.5*r, np.sqrt(3)/2*r])
        d3 = np.asarray([1.5*r, -np.sqrt(3)/2*r])
        ref_dirs = [[1,0,0],[0,1,0],[0,0,1],
                [-1,0,0],[0,-1,0],[0,0,-1]]
        v = np.asarray(ref_dirs[int(np.floor(lid/ring))])*ring
        rem = np.mod(lid,ring)
        pos = int(np.mod(np.floor(lid/(ring)),3))
        while rem:
            if not v[pos]:
                pos = pos + 1 if pos<2 else 0
            if pos<2:
                if v[pos] < 0:
                    v[pos] +=1
                    v[pos+1] -=1
                else:
                    v[pos] -= 1
                    v[pos+1] +=1
            else:
                if v[pos] < 0:
                    v[pos] +=1
                    v[0] +=1
                else:
                    v[pos] -= 1
                    v[0] -=1
            rem-=1
        return d1*v[0] + d2*v[1] + d3*v[2]

    def _get_proj_omm_loc(self, ring, lid):
        x,y = self._get_cartesian_omm_loc(ring, lid)
        return AlbersProjectionMap(self._reye).invmap(x, y)
    
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
                    relative_neighborpos = i + 1
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

    def get_neighbors(self, config = {'rule': 'superposition'}):
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
        rule = self._getconfig(config, 'rule', 'superposition')

        if rule == 'superposition':
            return self._supneighborslist
        elif rule == 'adjacency':
            return self._adjneighborslist
        else:
            valid_values = ['superposition', 'adjacency']
            raise ValueError("rule attribute must be one of %s" % ", "
                                 .join(str(x) for x in valid_values))

    def get_positions(self, config={'coord': 'spherical', 'include': 'center',
                                    'add_dummy': True}):
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
        coord = self._getconfig(config, 'coord', 'spherical')
        include = self._getconfig(config, 'include', 'center')
        add_dummy = self._getconfig(config, 'add_dummy', True)

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
        elif include == 'center':
            for ommatidium in self._ommatidia:
                if add_dummy or not ommatidium.is_dummy():
                    eyelat, eyelong = ommatidium.get_eyepoint()
                    latitudes.append(eyelat)
                    longitudes.append(eyelong)
        else:
            valid_values = ['all', 'R1toR6', 'center']
            raise ValueError("include attribute must be one of %s" % ", "
                             .join(str(x) for x in valid_values))

        if coord == 'spherical':
            return (np.array(latitudes), np.array(longitudes))
        elif coord == 'cartesian3D':
            nplatitudes = np.array(latitudes)
            nplongitudes = np.array(longitudes)
            sinlats = np.sin(nplatitudes)
            coslats = np.cos(nplatitudes)
            sinlongs = np.sin(nplongitudes)
            coslongs = np.cos(nplongitudes)
            return (sinlats*coslongs, sinlats*sinlongs, coslats)
        elif coord == 'cartesian2D':
            nplatitudes = np.array(latitudes)
            nplongitudes = np.array(longitudes)
            r2d = nplatitudes/(PI/2)
            sinlongs = np.sin(nplongitudes)
            coslongs = np.cos(nplongitudes)
            return (r2d*coslongs, r2d*sinlongs)
        else:
            valid_values = ['spherical', 'cartesian3D', 'cartesian2D']
            raise ValueError("coord attribute must be one of %s" % ", "
                             .join(str(x) for x in valid_values))


    def get_intensities(self, file=None, config={}):
        """ If file is not None then intensities are generated from the file
            otherwise they are generated from information in configuration
        """
        # the order of ommatidia in output is the exact order of
        # _ommatidia parameter
        if file is None:
            return self.get_intensities_from_generatedimage(config)
        else:
            return self.get_intensities_from_fileimage(file, config)

    # TODO remove repeated parts
    def get_intensities_from_generatedimage(self, config={}):
        """ generates input image programmatically
            config: {
                'type': 'bar'
                        'steps': number of simulation steps, 
                                 also determines the size of output
                        'dt':    needed for the correct scaling of speed (s)
                        'width': width of the bar (pixels)
                        'dir':   0 top to bottom, 1 left to right
                        'speed': how fast the bar is moving (pixels/s)
                        'shape': image dimensions
                        'levels': tuple (default value, bar value)
                'type': 'grating'
                        'steps': number of simulation steps, 
                                 also determines the size of output
                        'dt':    needed for the correct scaling of speed (s)
                        'x_freq': frequency in x direction (higher -> more bars)
                        'y_freq': frequency in y direction (higher -> more bars)
                        'x_speed': speed that the grating is moving in x dir
                                   (pixels/s)
                        'y_speed': speed that the grating is moving in y dir
                                   (pixels/s)
                        'shape': image dimensions
                        'sinusoidal': if True grating is a sine function
                                      otherwise only the sign of this function 
                                      is computed
                        'levels': if sinusoidal is True these are the limits of
                                  the sine function (low, high)
                'type': 'ball'
                        'steps': number of simulation steps, 
                                 also determines the size of output
                        'dt':    needed for the correct scaling of speed (s)
                        'center': center of ball
                        'speed': speed of expansion (pixels/s)
                        'white_in_side': if True ball is black(high value) and 
                                         side is white(low value) 
                                         and conversely
                        'shape': image dimensions
                        'levels': (low value, high value)
                'output_file':<filename>
            }
        """
        output_file = self._getconfig(config, 'output_file', None)
        type = self._getconfig(config, 'type', 'bar')
        # same for all
        dt = self._getconfig(config, 'dt', 1e-4)
        time_steps = self._getconfig(config, 'steps', 1000)
        shape = self._getconfig(config, 'shape', (100, 100))
        levels = self._getconfig(config, 'levels', (0, 85))

        intensities = np.empty((time_steps, 6*self._nommatidia),
                                dtype='float32')
        image = np.empty((time_steps,) + shape,
                                dtype='float32')


        # screen positions
        # should be much more than the number of photoreceptors
        # shape (M, P)*self._nrings
        M = self.MERIDIANS
        P = self.PARALLELS
        screenlat, screenlong = np.meshgrid(np.linspace((PI/2)/P,
                                                        PI/2, P),
                                            np.linspace(-PI, PI*(M-2)/M, M))

        mapscreen = self.OMMATIDIUM_CLS.MAP_SCREEN
        # plane positions
        # shape (M, P)*self._nrings
        mx, my = mapscreen.map(screenlat, screenlong)   # map from spherical
                                                        # grid to plane

        # photoreceptor positions
        photorlat, photorlong = self.get_positions({'coord': 'spherical',
                                                    'include': 'R1toR6',
                                                    'add_dummy': False})


        self._pre_get_intensities(mx.shape[0], mx.shape[1],
                                  photorlat,photorlong,
                                  screenlat,screenlong)
        mx = mx - mx.min()  # start from 0
        my = my - my.min()
        
        if type == 'bar':
            width = self._getconfig(config, 'width', 20)
            dir = self._getconfig(config, 'dir', 0)
            speed = self._getconfig(config, 'speed', 500)

            for i in range(time_steps):
                image[i] = np.ones(shape, dtype=np.double)*levels[0]
                if dir == 1:
                    st = int(np.mod(i*speed*dt, shape[1]))
                    en = min(int(st+width), shape[1])
                    image[i,:,st:en] = levels[1]
                else:
                    st = int(np.mod(i*speed*dt, shape[0]))
                    en = min(int(st+width), shape[0])
                    image[i,st:en, :] = levels[1]
                
                # photons were stored for 1ms
                image[i] *= (dt/1e-3)
                intensities[i] = self._get_intensities_from_image(image[i], mx, my)
        elif type == 'grating':
            x_freq = self._getconfig(config, 'x_freq', 0.05)
            y_freq = self._getconfig(config, 'y_freq', 0)
            x_speed = self._getconfig(config, 'x_speed', 500)
            y_speed = self._getconfig(config, 'y_speed', 0)
            sinusoidal = self._getconfig(config, 'sinusoidal', True)

            x, y = np.meshgrid(np.arange(float(shape[1])),
                               np.arange(float(shape[0])))
            if sinusoidal:
                sinfunc = lambda x: ((np.sin(x)+1)/2)*(levels[1]-levels[0]) + \
                                    levels[0]
            else:
                sinfunc = lambda x: np.sign(np.sin(x))*((levels[1]-levels[0])/2) \
                                    + (levels[1]+levels[0])/2

            for i in range(time_steps):
                image[i] = (sinfunc(x_freq*2*np.pi*(x - x_speed*i*dt)) + 
                        sinfunc(y_freq*2*np.pi*(y - y_speed*i*dt)))
                # photons were stored for 1ms
                image[i] *= (dt/1e-3)
                intensities[i] = self._get_intensities_from_image(image[i], mx, my)

        elif type == 'ball':
            center = self._getconfig(config, 'center', None)
            speed = self._getconfig(config, 'speed', 100)
            white_in_side = self._getconfig(config, 'white_in_side', True)

            x, y = np.meshgrid(np.arange(float(shape[1])),
                               np.arange(float(shape[0])))

            for i in range(time_steps):
                image[i] = (-1 if white_in_side else 1) * np.sign(
                        np.sqrt((x-center[0])**2+(y-center[1])**2)
                        - i*self._dt*self._speed).astype(np.double)
                image[i] = image*((levels[1]-levels[0])/2)+(levels[1]+levels[0])/2
                # photons were stored for 1ms
                image[i] *= (dt/1e-3)
                intensities[i] = self._get_intensities_from_image(image[i], mx, my)
        else:
            raise ValueError("Invalid type {}, should be ball, bar or grating"
                             .format(type))

        if output_file:
            # intensities will be stored in 2 files if output file is set
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('array', intensities.shape,
                                 dtype=np.float64,
                                 data=intensities)
        with h5py.File(self.INTENSITIES_FILE, 'w') as f:
            f.create_dataset('array', intensities.shape,
                             dtype=np.float64,
                             data=intensities)
        with h5py.File(self.IMAGE_FILE, 'w') as f:
            f.create_dataset('array', image.shape,
                             dtype=np.float32,
                             data=image)

    def get_intensities_from_fileimage(self, file, config={}):
        """ file: input image file, mat file is assumed with a variable im in it
            config: {'type': 'image'(default)/'video',
                     'steps':<simulation steps>,
                     'dt':<time step>,
                     'output_file':<filename>}
            returns: numpy array with with height the number of simulation
                     steps and width the number of neurons.
                     The order of neurons is the same as that returned
                     in get_positions, with 'include'
                     configuration parameter set to 'R1toR6'
        """
        dt = self._getconfig(config, 'dt', 1e-4)
        time_steps = self._getconfig(config, 'steps', 1000)
        type = self._getconfig(config, 'type', 'image')
        output_file = self._getconfig(config, 'output_file', None)

        if output_file in [self.POSITIONS_FILE, 
                       self.IMAGE_FILE]:
            raise ValueError('Invalid value for intensities file: {}'
                         ', filename is reserved for another purpose'
                         .format(output_file))

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
        # shape (M, P)*self._nrings
        M = self.MERIDIANS
        P = self.PARALLELS
        screenlat, screenlong = np.meshgrid(np.linspace((PI/2)/P,
                                                        PI/2, P),
                                            np.linspace(-PI, PI*(M-2)/M, M))

        mapscreen = self.OMMATIDIUM_CLS.MAP_SCREEN
        # plane positions
        # shape (M, P)*self._nrings
        mx, my = mapscreen.map(screenlat, screenlong)   # map from spherical
                                                        # grid to plane

        # photoreceptor positions
        photorlat, photorlong = self.get_positions({'coord': 'spherical',
                                                    'include': 'R1toR6',
                                                    'add_dummy': False})


        self._pre_get_intensities(mx.shape[0], mx.shape[1],
                                  photorlat,photorlong,
                                  screenlat,screenlong)
        if type == 'image':
            mx = mx - mx.min()  # start from 0
            my = my - my.min()
            mx *= h_im/mx.max()  # scale to image size
            my *= w_im/my.max()

            # shape (M, P)*self._nrings
            transimage = transform.interpolate((mx, my))
            intensities = self._get_intensities(transimage, self.ind_weights)
            intensities = np.tile(intensities, (time_steps, 1))

            positions = None
        elif type == 'video':
            intensities = np.empty((time_steps, len(photorlat)),
                                   dtype='float32')
            positions = np.empty((time_steps, 4),
                                   dtype='float32')
            mx = mx - mx.min()  # start from 0
            my = my - my.min()
            for i in range(time_steps):
                scale = 0.1
                mx = mx + scale*(2*np.random.random() - 1)  # move randomly
                my = my + scale*(2*np.random.random() - 1)  # between -1 and 1
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

                intensities[i] = self._get_intensities(transimage, self.ind_weights)
                positions[i] = [mx.min(), mx.max(), my.min(), my.max()]

        if output_file:
            # intensities will be stored in 2 files if output file is set
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('array', intensities.shape,
                                 dtype=np.float64,
                                 data=intensities)

        with h5py.File(self.INTENSITIES_FILE, 'w') as f:
            f.create_dataset('array', intensities.shape,
                             dtype=np.float64, data=intensities)
        if positions is not None:
            #TODO
            with h5py.File(self.POSITIONS_FILE, 'w') as f:
                f.create_dataset('array', positions.shape,
                                 dtype=np.float32, data=positions)
            for i, position in enumerate(positions):
                print i
                assert(positions[i-1,1]-positions[i-1,0] == positions[i,1]-positions[i,0])

            with h5py.File(self.IMAGE_FILE, 'w') as f:
                f.create_dataset('array', image.shape,
                                 dtype=np.float32, data=image)
        else:
            image = np.tile(image, (time_steps, 1, 1))
            with h5py.File(self.IMAGE_FILE, 'w') as f:
                f.create_dataset('array', image.shape,
                                 dtype=np.float32, data=image)

        return intensities

    def _get_intensities_from_image(self, image, mx, my):
        h_im, w_im = image.shape
        transform = ImageTransform(image)

        mx *= h_im/mx.max()  # scale to image size
        my *= w_im/my.max()

        # shape (M, P)*self._nrings
        transimage = transform.interpolate((mx, my))
        return self._get_intensities(transimage, self.ind_weights)

    def _get_intensities(self, transimage, ind_weights):
        intensities = np.empty(len(ind_weights), dtype='float32')
        for i, (indlong, indlat, weights) in enumerate(ind_weights):
            try:
                pixels = transimage[indlong, indlat]  # nxn np array
            except IndexError:
                print('Array size is {}'.format(transimage.shape))
                print('Indexes are {} and {}'.format(indlat, indlong))
                raise

            # works with >=1.8 numpy
            intensities[i] = np.sum(pixels*weights)

        return intensities

    def _pre_get_intensities(self, h_tim, w_tim, photorlat, photorlong,
                             screenlat, screenlong):
        dlat = screenlat[0, 1] - screenlat[0, 0]
        dlong = screenlong[1, 0] - screenlong[0, 0]

        ind_weights = [0] * len(photorlat)
        for i, (lat, long) in enumerate(zip(photorlat, photorlong)):

            # position in grid (float values)
            parallelf = (w_tim - 1)*lat/(PI/2)  # float value
            meridianf = (h_tim - 1)*(long + PI)/(2*PI)  # float value

            # for each photoreceptor point find indexes of
            # nxn closest points on grid (int values)
            indlat, indlong = self.get_closest_indexes(parallelf, meridianf,
                                                       min1=0, min2=0,
                                                       max1=w_tim, max2=h_tim,
                                                       n=self.get_min_points())
            weights = self.get_gaussian_sphere(screenlat[indlong, indlat],
                                               screenlong[indlong, indlat],
                                               lat, long, dlat, dlong,
                                               self.get_kappa())
            ind_weights[i] = (indlong, indlat, weights)

        self.ind_weights = ind_weights

    def get_kappa(self):
        try:
            return self._kappa
        except AttributeError:
            eps = 1e-4
            # value of inner product/distance that will have eps probability
            inprc = math.cos(self.get_min_angle())
            # looking for largest kappa such that
            # [kappa/(2pi(e^kappa-e^(-kappa)))] * exp(kappa*inprc) < eps
            # e^(-kappa) can be ignored since kappa will be large so
            # x*exp(x) < 2*pi*eps*(inprc-1)
            # where x=(inprc-1)*kappa

            self._kappa = self.compute_W(2*PI*eps*(inprc-1))/(inprc-1)
            print('Kappa: {}'.format(self._kappa))
            return self._kappa

    def get_min_angle(self):
        return (PI/2)/self._nrings

    def get_min_points(self):
        return self.PARALLELS//self._nrings + 1

    @staticmethod
    def compute_W(x):
        """
            computes value w such that w*e^w = x using Newtons numerical method
        """
        assert x>=-1
        if x < 0:
            w = -2
        else:
            w = x
        while True:
            ew = math.exp(w)
            wNew = w - (w*ew - x) / (w * ew + ew)
            if abs(w - wNew) < 1e-8: break
            w = wNew
        return w

    def get_closest_indexes(self, f1, f2, min1, min2, max1, max2, n):
        """ Given a point (f1, f2) return n closest points to f1 in the box
            [min1, max1)
            for f2 all indices from min2 to max2 are returned but it is kept
            for consistency with the previous version
        """
        ind1 = np.linspace(np.floor(f1) + (1-n/2), np.ceil(f1) + (-1+n/2), n) \
            .astype(int)
        ind1 = np.minimum(ind1, max1-1)
        ind1 = np.maximum(ind1, min1)

        ind2 = np.linspace(min2, max2-1, max2-min2).astype(int)
        return np.meshgrid(ind1, ind2)

    @staticmethod
    def get_gaussian_sphere(lats, longs, reflat, reflong, dlat, dlong, kappa):
        """ Computes gaussian function on sphere at points (lats, longs),
            with a given center (reflat, reflong).

            see also:
            http://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
        """

        # inner product of reference point with other points translated to
        # spherical coordinates
        in_prod = np.sin(lats)*np.sin(reflat)*np.cos(reflong-longs) \
            + np.cos(lats)*np.cos(reflat)
        #approximate e^kappa - e^-kappa as e^-kappa as kappa is generally large
        func3 = (kappa/(2*PI))*np.exp(kappa*(in_prod-1))*np.sin(lats)*dlat*dlong
        return func3

    def plot_ommatidia(self, fname):
        fig = plt.figure(figsize=plt.figaspect(1), dpi = 80)
        
        latpositions, longpositions = self.get_positions(
                                {'coord': 'spherical', 'include': 'R1',
                                 'add_dummy': False})
        U, V = np.mgrid[0:0.8*np.pi/2:complex(0, 3000),
                        0:2*np.pi:complex(0, 3000)]
        X = np.cos(V)*np.sin(U)
        Y = np.sin(V)*np.sin(U)
        Z = np.cos(U)
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        xx = np.cos(longpositions) * np.sin(latpositions)
        yy = np.sin(longpositions) * np.sin(latpositions)
        zz = np.cos(latpositions)
        X_shape = X.shape

        ax2 = fig.add_subplot(1, 1, 1, projection='3d')
        
        ax2.xaxis.set_ticks([])
        ax2.yaxis.set_ticks([])
        ax2.zaxis.set_ticks([])

        C = griddata((xx,yy,zz), np.mod(np.arange(len(xx)),25)*10, (x,y,z),
                     'nearest').reshape(Z.shape)
        C = cm.Paired(C)
        ax2.plot_surface(X, Y, Z,facecolors=C, antialiased=False, shade=False)
        fig.canvas.draw()
        fig.savefig(fname)

    def _get_lamina_ids(self, neuron_name):
        G = self._lamina_graph
        neurons = G.node.items()

        ids = []
        for id, neu in neurons:
            if neuron_name == neu['name']:
                ids.append(id)
        return ids

    # TODO KP is use of -1 and 1 consistent? use only one of them maybe?
    def connect_retina_lamina(self, manager, ret_lpu, lam_lpu, from_file=False):
        if first_lpu > LPU_ORDER['r'] or last_lpu < LPU_ORDER['l']:
            return
        if not from_file:
            #.values or .tolist()
            print('Initializing selectors')
            # some workarounds
            ret_sel = ','.join(['/retout/'+ str(sel[-1]) for sel in ret_lpu.interface.index.tolist()])
            lam_sel = ','.join(['/' + str(sel[0]) + '/' + str(sel[1]) for sel in lam_lpu.interface.index.tolist()])
            
            print('Initializing pattern with selectors')
            pat = Pattern(ret_sel, lam_sel)
            ret_sel = ','.join(['/retout/'+ str(sel[-1]) for sel in ret_lpu.interface.index.tolist()])
            lam_sel_in = ','.join(['/' + str(sel[0]) + '/' + str(sel[-1])
                                   for sel in lam_lpu.interface.index.tolist()
                                   if str(sel[0])=='retin' or str(sel[0])=='medin'])
            lam_sel_out = ','.join(['/lamout/' + str(sel[-1])
                                    for sel in lam_lpu.interface.index.tolist()
                                    if not str(sel[0]) == 'retin' and not str(sel[0])=='medin'])
            print('Setting selector attributes in pattern')
            pat.interface[ret_sel] = [0, 'in', 'gpot']
            pat.interface[lam_sel_in] = [1, 'out', 'gpot']
            pat.interface[lam_sel_out] = [1, 'in', 'gpot']

            for sel in lam_sel_in.split(','):
                if str(sel).startswith('/retin'):
                    pat[sel.replace('in', 'out'), sel] = 1
            pat_file = open(self.RET_LAM_PAT_FILE, 'wb')
            pickle.dump(pat, pat_file)
            pat_file.close()
        else:
            pat_file = open(self.RET_LAM_PAT_FILE, 'rb')
            pat = pickle.load(pat_file)
            pat_file.close()
        
        print('Connecting LPUs with the pattern')
        print(pat)
        manager.connect(ret_lpu, lam_lpu, pat, 0, 1)

    def connect_lamina_medulla(self, manager, lam_lpu, med_lpu, from_file=False):
        if first_lpu > LPU_ORDER['l'] or last_lpu < LPU_ORDER['m']:
            return
        if not from_file:
            #.values or .tolist()
            print('Initializing selectors')
            # some workarounds
            lam_sel = ','.join(['/' + str(sel[0]) + '/' +  str(sel[1]) for sel in lam_lpu.interface.index.tolist()])
            med_sel = ','.join(['/' + str(sel[0]) + '/' +  str(sel[1]) for sel in med_lpu.interface.index.tolist()])
            
            print('Initializing pattern with selectors')
            
            pat = Pattern(lam_sel, med_sel)
            lam_sel_in = ','.join(['/' + str(sel[0]) + '/' + str(sel[-1])
                                   for sel in lam_lpu.interface.index.tolist()
                                   if str(sel[0])=='retin' or str(sel[0])=='medin'])
            lam_sel_out = ','.join(['/lamout/' + str(sel[-1])
                                    for sel in lam_lpu.interface.index.tolist()
                                    if not str(sel[0]) == 'retin' and not str(sel[0])=='medin'])
            med_sel_in = ','.join(['/' + str(sel[0]) + '/' + str(sel[-1])
                                   for sel in med_lpu.interface.index.tolist()
                                   if str(sel[0])=='retin' or str(sel[0])=='lamin'])
            med_sel_out = ','.join(['/medout/' + str(sel[-1])
                                    for sel in med_lpu.interface.index.tolist()
                                    if not str(sel[0]) == 'retin' and not str(sel[0])=='lamin'])
            
            
            print('Setting selector attributes in pattern')
            pat.interface[med_sel_out] = [1, 'in', 'gpot']
            pat.interface[med_sel_in] = [1, 'out', 'gpot']
            pat.interface[lam_sel_in] = [0, 'out', 'gpot']
            pat.interface[lam_sel_out] = [0, 'in', 'gpot']

            for sel in med_sel_in.split(','):
                if str(sel).startswith('/lamin'):
                    pat[sel.replace('in', 'out'), sel] = 1

            for sel in lam_sel_in.split(','):
                if str(sel).startswith('/medin'):
                    pat[sel.replace('in', 'out'), sel] = 1

            pat_file = open(self.LAM_MED_PAT_FILE, 'wb')
            pickle.dump(pat, pat_file)
            pat_file.close()
        else:
            pat_file = open(self.LAM_MED_PAT_FILE, 'rb')
            pat = pickle.load(pat_file)
            pat_file.close()
        
        print('Connecting LPUs with the pattern')
        print(pat)
        manager.connect(lam_lpu, med_lpu, pat, 0, 1)

    
    def _generate_retina(self, retina_only=False):
        G = nx.DiGraph()

        photoreceptor_num = 6*self._nommatidia
        
        latpositions, longpositions = self.get_positions(
            {'coord': 'spherical', 'include': 'R1toR6',
             'add_dummy': False})
        print len(latpositions)
        for i in range(photoreceptor_num):
            G.node[i] = {
                'model': 'Photoreceptor',
                'name': 'R' + str(int(np.mod(i,6)) + 1),
                'extern': True,  # gets input from file
                'public': False,  # True if it's an output neuron
                'spiking': False,
                'num_microvilli': 30000,
                'lat': float(latpositions[i]),
                'long': float(longpositions[i])
            }
            if not retina_only:
                G.node[i]['selector'] = '/retout/' + str(i)
                G.node[i]['public'] = True
        self._retina_graph = G

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
                    neuron.update_selector('/retin/' + str(photor_id))

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

    def _generate_neurons(self, n_list, n_dict, cartridge_num, LPU='lam'):
        if LPU == 'lam':
            neuron_types = model_lamina.NEURON_LIST
        else:
            neuron_types = model_medulla.NEURON_LIST
        alpha_profiles = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

        latpositions, longpositions = self.get_positions(
            {'coord': 'spherical', 'include': 'center',
             'add_dummy': False})

        for neuron_params in neuron_types:
            neuron_name = neuron_params['name']
            if neuron_name=='Am':
                am_num = neuron_params['num']
                bound = PI/np.sqrt(am_num)
                am_lats = (PI/2)*np.random.random(am_num)
                am_longs = 2*PI*np.random.random(am_num) - PI
                
                # create amacrine neurons
                for i in range(am_num):
                    n = Neuron(neuron_params)
                    n.add_param('lat', float(am_lats[i]))
                    n.add_param('long', float(am_longs[i]))
                    n.num = len(n_list)
                    n_list.append(n)
                    n_dict[(i, 'Am')] = n

                # connect them to cartridges
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
                    n.add_param('lat', float(latpositions[i]))
                    n.add_param('long', float(longpositions[i]))
                    if LPU == 'lam':
                        if neuron_params['output']:
                            n.update_selector('/lamout/' + neuron_name + '_' + str(i))
                        if neuron_params['model'] == PORT_IN_GPOT and neuron_name in model_lamina.MED_INPUTS:
                            n.update_selector('/medin/' + neuron_name + '_' + str(i))
                    else:
                        if neuron_params['output']:
                            n.update_selector('/medout/' + neuron_name + '_' + str(i))
                        if neuron_params['model'] == PORT_IN_GPOT:
                            n.update_selector('/lamin/' + neuron_name + '_' + str(i))
                    n_list.append(n)
                    n_dict[(i, neuron_name)] = n

    def _generate_synapses(self, s_list, n_dict, cartridge_num, LPU='lam'):
        if LPU=='lam':
            synapse_types = model_lamina.SYNAPSE_LIST
        else:
            synapse_types = model_medulla.SYNAPSE_LIST
        
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

    def _generate_medulla(self):
        G = nx.MultiDiGraph()

        # ommatidia are equal with the cartridges
        cartridge_num = self._nommatidia

        # lists will have only one copy of
        # neurons synapses
        # dicts may have duplicates
        neuron_list = []
        synapse_list = []
        neuron_dict = {}

        self._generate_neurons(neuron_list, neuron_dict, cartridge_num, LPU='med')
        
        for (count,node) in enumerate(neuron_list):
            node.num = count
            
        self._generate_synapses(synapse_list, neuron_dict, cartridge_num, LPU='med')
        
        # add nodes to graph
        for node in neuron_list:
            # if neuron is not dummy
            if not node.is_dummy():
                G.add_node(node.num, node.params)

        for edge in synapse_list:
            edge.process_before_export()
            G.add_edge(edge.prenum, edge.postnum, attr_dict=edge.params)

        self._medulla_graph = G


    def write_retina(self, output_file):
        if first_lpu <= LPU_ORDER['r'] and last_lpu >= LPU_ORDER['r']:
            nx.write_gexf(self._retina_graph, output_file)

    def write_lamina(self, output_file):
        if first_lpu <= LPU_ORDER['l'] and last_lpu >= LPU_ORDER['l']:
            nx.write_gexf(self._lamina_graph, output_file)

    def write_medulla(self, output_file):
        if first_lpu <= LPU_ORDER['m'] and last_lpu >= LPU_ORDER['m']:
            nx.write_gexf(self._medulla_graph, output_file)

    def _getconfig(self,config, key, default):
        try:
            return config[key]
        except KeyError:
            return default
