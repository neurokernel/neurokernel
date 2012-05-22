import pycuda.gpuarray as garray
import pycuda.driver as cuda
import numpy as np

class Conectivity:

    def __init__(self):
        self.modules

    def add_module(self, module):
        self.modules.append(module)

    def rm_module(self, module):
        self.modules.remove(module)

    def add_connectivity(self, connectivity):
        self.connectivities.append(connectivity)

    def rm_connectivity(self, connecivity):
        self.connectivities.remove(connecivity)
