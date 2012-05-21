import pycuda.driver as cuda
import Module
import numpy as np
import tools.parray as parray
import progressbar as pb
import Manager

dev1 = 1

cuda.init()

np.random.seed(0)

dt = 1e-4
dur = 1.0
Nt = int(dur / dt)
t = np.arange(0, 1, dt)

manager = Manager.manager()
manager.add_module(Module.Module(manager, dt, num_in_non, num_in_spike,
                                 num_proj_non, num_proj_spike, dev1))

#input video
I_ext = parray.to_gpu(np.ones([dur / dt, 4608]))

out = np.empty((Nt, m1.network.num_neurons), np.double)

playstep = 100
pbar = pb.ProgressBar(maxval = Nt).start()
for i in range(Nt):
    pbar.update(i)
    manager.modules[0].run_step(int(I_ext.gpudata) + I_ext.dtype.itemsize * \
                                I_ext.ld * i, None, out[i, :], None)

pbar.finish()
