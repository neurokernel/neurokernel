import atexit
import pycuda.gpuarray as garray
import pycuda.driver as cuda
import Network as nn

"""
Can you please construct an object-oriented PyCUDA implementation of a network
of ML neurons randomly connected by alpha function synapses (similar to the
IAF network you implemented for E9070) that we can use for testing the
architecture?
I imagine that a standard numerical ODE solver such as low-order Runge-Kutta
should be sufficient for simulating the network.
Also, I'm not sure what dynamics we should expect to observe for such a
network. You may need to talk to Yiyin or Nikul for further information about
both of these points.
"""
class Module:

    def __init__(self, in_non_list, in_spike_list, proj_non, proj_spike,
                 param, dev):
        """
        Interface between LPU and architecture.
            Parameters
            ----------
            in_non_list : list of numpy.ndarray of numpy.float64
                States (membrane voltages) of external non-spiking neurons
                presynaptic to the module at the current time.
            in_spike_list : list of numpy.ndarray of numpy.int32
                Indices of external spiking neurons presynaptic to the module
                that emitted a spike at the current time.
            pro_non : numpy.ndarray of numpy.float64
                States (membrane voltages) of non-spiking projec- tion neurons
                at current time.
            proj_spike : numpy.ndarray of numpy.int32
                Indices of spiking projection neurons that emitted a spike at
                the current time.
            param : list
                List of variables to configure the LPU.
            dev : int
                Indicates which GPU device will be used by this module.
        """

        self.in_non_list = in_non_list
        self.in_spike_list = in_spike_list
        self.proj_non = proj_non
        self.proj_spike = proj_spike

        ctx = cuda.Device(dev).make_context()
        atexit.register(ctx.pop)

        self.network = nn.Network(in_non_list, in_spike_list, proj_non,
                                  proj_spike, param)

    def run_step(self, I_ext, out):

        self.network.run_step(I_ext, out)
