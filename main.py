import pycuda.driver as cuda
import Module
import numpy as np
import tools.parray as parray
import progressbar as pb

dev1 = 1

cuda.init()

np.random.seed(0)

dt = 1e-4
dur = 1.0
Nt = int(dur / dt)
t = np.arange(0, 1, dt)

m1 = Module.Module(dt, dev1)

#input video
I_ext = parray.to_gpu(np.ones([dur / dt, 4608]))

out = np.empty((Nt, m1.network.num_neurons), np.double)

playstep = 100
pbar = pb.ProgressBar(maxval = Nt).start()
for i in range(Nt):
    pbar.update(i)
    m1.run_step(int(I_ext.gpudata) + I_ext.dtype.itemsize * I_ext.ld * i, None,
                     out[i, :], None)

pbar.finish()
