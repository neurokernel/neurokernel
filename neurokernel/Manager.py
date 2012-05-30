class Manager (object):

    def __init__(self):

        self.modules = []
        self.connectivities = []

    def add_module(self, module):
        self.modules.append(module)

    def rm_module(self, module):
        self.modules.remove(module)

    def add_connectivity(self, connectivity):
        self.connectivities.append(connectivity)

    def rm_connectivity(self, connecivity):
        self.connectivities.remove(connecivity)

    def start(self):

        for m in self.modules:
            m.start()
