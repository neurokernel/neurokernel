import neurokernel.Manager as Manager
from neurokernel.MockSystem.MockSystem import MockSystem

manager = Manager.Manager()
manager.add_module(MockSystem(manager, 768, 8, 1e-4, 4608, 0, 4608, 0, 1))

manager.start()
