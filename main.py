import neurokernel.Manager as Manager
import neurokernel.Mock.MockNetwork as NN

manager = Manager.Manager()
manager.add_module(NN(manager, dt = 1e-4, num_in_non = 4608, num_in_spike = 0,
                      num_proj_non = 4608, num_proj_spike = 0, device = 1))

manager.start()
