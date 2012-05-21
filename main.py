import Manager
import Module

manager = Manager.Manager()
manager.add_module(Module.Module(manager, 1e-4, 4608, 0, 4608, 0, 1))

manager.start()
