import neurokernel.Manager as Manager
from neurokernel.MockSystem.MockSystem import MockSystem

manager = Manager.Manager()
manager.add_module(MockSystem(manager, 1024, 8, 1e-4, 4608, 0, 4608, 0, 0))
manager.add_module(MockSystem(manager, 2048, 8, 1e-4, 4608, 0, 4608, 0, 1))

manager.start()
