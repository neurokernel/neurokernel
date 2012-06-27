"""This is the Module.py class. This class comprises a brain processing unit,
which means that this class can be from a LPU to an entire sensory system.

Known issues
------------
    - This class do not support dynamic modifications yet, which means that it
    is not possible change the mapping between two modules in run-time.

"""
import atexit
import pycuda.driver as cuda
from neurokernel.tools import parray
from multiprocessing import Process
import numpy as np
import logging

class Module (Process):
    """
    A module comprises one or more local processing units and it is the
    interface between those LPU and the manager.

    Attributes
    ----------
    manager : neurokernel.Manager
        Module manager that manages this module instance.
    dt : numpy.double
        Time resolution of the simulation.
    inputs : array_like
        List with the number of neurons per type states (membrane voltages)
        of external non-spiking neurons presynaptic to the module at
        the current time.
    outputs : array_like
        Number of non-spiking neuron's states (membrane voltages) of
        non-spiking projec- tion neurons at current time.
    in_conn : array_like
        Comprises a list with all connectivity modules for incoming connections.
    device : int
        GPU device used by the module instance.

    See Also
    --------
    Manager.Manager : Class that manages Modules and Connectivities.

    """

    def __init__(self, manager, dt, num_proj, device):
        """
        A module comprises one or more local processing units and it is the
        interface between those LPU and the manager.

        Parameters
        ----------
        manager : neurokernel.Manager
            Module manager that manages this module instance.
        dt : numpy.double
            Time resolution.
        num_proj : array_like
            Number of non-spiking neuron's states (membrane voltages) of
            non-spiking projection neurons at current time.
        device : int
            GPU device used by the module instance.

        Examples
        --------
        In this example it's created  a module with two types of projection
        neurons: 6 neurons type1 and 3 neurons type2 as output.
        >>> ...
        >>> mod1 = Module(manager, 1e-5, [6, 3], 0)
        >>> print mod1.outputs
        [array([[ 0.,  0.,  0.,  0.,  0.,  0.]]), array([[ 0.,  0.,  0.]])]
        >>> ...

        See Also
        --------
        neurokernel.Manager : Class that manages Modules and Connectivities.

        """

        Process.__init__(self)

        self.manager = manager
        self.running = True #temp
        self.dt = dt
        self.device = device

        self.outputs = [np.zeros([1, x], dtype = np.float64) for x in num_proj]

        # Connectivity
        self.in_conn = []
        self.out = []

    def init_gpu(self):
        """
        Code to run after CUDA device initialization

        Since a CUDA device is initialized in the run() method (i.e.,
        when the process is forked) rather than the constructor,
        initialization code that should be run before the module
        simulation begins should be included in this method.
        """

        pass

    def run_step(self):
        """
        Run one step of the module simulation.

        Parameters
        ----------
        in_list : list of pycuda.gpuarray.GPUArray
            States of external input neurons divided by type. Each element of
            this list is a vector with neuron states of one type.
        out_list : list of pycuda.gpuarray.GPUArray
            States of output neurons divided by type. Each element of this list
            is a vector with neuron states of one type.

        Raises
        ------
        NotImplementedError
            You cannot run this method on the base class.

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
        """
        Body of process.

        """

        # Initialize CUDA context:
        cuda.init()
        ctx = cuda.Device(self.device).make_context()
        atexit.register(ctx.pop)

        # Pre-simulation initialization that requires a valid GPU context:
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
            temp = int(I_ext.gpudata) + I_ext.dtype.itemsize * I_ext.ld * i
            self.run_step([temp], [out[i, :]])

#        while(self.running):
#            self.__run_step(self.in_non_list, self.in_spike_list, proj_non,
#                       proj_spike)
#            self.proj_non.append(proj_non)
#            self.proj_spike.append(proj_spike)
#            __sync()
