import atexit
import pycuda.driver as cuda
from neurokernel.tools import parray
from multiprocessing import Process
import numpy as np

class Module (Process):

    def __init__(self, manager, dt, num_in_non, num_in_spike, num_proj_non,
                 num_proj_spike, device):
        """
        A module comprises one or more local processing units and it is the
        interface between those LPU and the manager.
            Parameters
            ----------
            manager : neurokernel.Manager
                See Manager.py for more information
            dt : numpy.double
            num_in_non : int
                Number of non-spiking neuron's states (membrane voltages)
                of external non-spiking neurons presynaptic to the module at
                the current time.
            num_in_spike : int
                Number of non-spiking neuron's indices of external spiking
                neurons presynaptic to the module that emitted a spike at the
                current time.
            num_proj_non : int
                Number of non-spiking neuron's states (membrane voltages) of
                non-spiking projec- tion neurons at current time.
            num_proj_spike : int
                Number of non-spiking neuron's indices of spiking projection
                neurons that emitted a spike at the current time.
            device : int
                Indicates which GPU device will be used by this module.
        """

        Process.__init__(self)

        self.manager = manager
        self.running = True
        self.dt = dt
        self.device = device

        self.num_in_non = num_in_non
        self.num_in_spike = num_in_spike

        self.proj_non = []
        self.proj_spike = []

    def init_gpu(self):
        """
        Because the module can run as an independent process, the CUDA driver
        is initialized in run-time and after the module has been initialized.
        However, some variables are stored in GPU and need a CUDA context ready.
        """

        raise NotImplementedError('You have to provide this method.')

    def run_step(self, in_non_list, in_spike_list, proj_non, proj_spike):
        """
        Runs one step of the module simulation.

        Parameters
            ----------
            in_non_list : list of
                See Manager.py for more information
            dt : numpy.double
            num_in_non : int
        """

        raise NotImplementedError('You have to provide this method.')

    def __sync(self):

        # receive input from outside
        I_ext = parray.to_gpu(np.ones([1, self.num_in_non]))
        self.in_non_list = int(I_ext.gpudata) + I_ext.dtype.itemsize
        self.in_spike_list = None

        # send output
        self.proj_non
        self.proj_spike

    def run(self):

        cuda.init()
        ctx = cuda.Device(self.device).make_context()
        atexit.register(ctx.pop)

        # If the system encapsulated by this module has any GPU initialization,
        # it must be invoked here.
        self.init_gpu()

        if self.num_in_non > 0:
            self.in_non_list = parray.to_gpu(np.ones([1, self.num_in_non]))
        else:
            self.in_non_list = None
        if self.num_in_spike > 0:
            self.in_spike_list = parray.to_gpu(np.ones([1, self.num_in_spike]))
        else:
            self.in_spike_list = None

#        proj_non = np.empty((1, len(self.proj_non)), np.double)
#        proj_spike = np.empty((1, len(self.proj_spike)), np.double)
        dt = self.dt

        I_ext = parray.to_gpu(np.ones([1 / dt, 4608]))
        out = np.empty((1 / dt, 4608), np.double)

        for i in range(int(1 / dt)):
            self.run_step(int(I_ext.gpudata) + \
                                        I_ext.dtype.itemsize * \
                                        I_ext.ld * i, None, out[i, :], None)

#        while(self.running):
#            self.__run_step(self.in_non_list, self.in_spike_list, proj_non,
#                       proj_spike)
#            self.proj_non.append(proj_non)
#            self.proj_spike.append(proj_spike)
#            __sync()
