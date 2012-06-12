import atexit, logger, signal
import numpy as np
import multiprocessing as mp
import pycuda.driver as cuda

from neurokernel.tools import parray
from neurokernel.tools.comm_utils import is_poll_in
from neurokernel.tools.gpu_utils import set_realloc

class Module (mp.Process):
    """
    A module comprises one or more local processing units and it is the
    interface between those LPU and the manager.

    Parameters
    ----------
    manager : neurokernel.Manager
        Module manager that manages this module instance.
    dt : numpy.double
        Time resolution.
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
        GPU device used by the module instance.

    See Also
    --------
    Manager.Manager
    
    """

    def __init__(self, dt, num_in_non, num_in_spike, num_proj_non,
                 num_proj_spike, device, *args, **kwargs):

        # Module identifier for zmq communication:
        self.id = kwargs.pop('id')

        # Port to use when connecting with the manager:
        self.port = kwargs.pop('port')

        self.logger = logging.getLogger('initializing module %s' % self.id)        
        Process.__init__(self)

        self.running = True
        self.dt = dt
        self.device = device

        self.num_in_non = num_in_non
        self.num_in_spike = num_in_spike

        self.proj_non = []
        self.proj_spike = []

        # List of connection objects:
        self.conn_list = []

    def init_zmq(self):
        """
        Initialize ZeroMQ.

        Notes
        -----
        This method must be called before the simulation commences.
        
        """

        self.ctx = zmq.Context()
        self.poller = zmq.Poller()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.IDENTITY, str(self.id))
        self.sock.connect("tcp://localhost:%i" % self.port)
        self.poller.register(self.sock, zmq.POLLIN)
        
    def init_gpu(self):
        """
        Initialize CUDA device.

        Notes
        -----
        Since a CUDA device is initialized in the run() method (i.e.,
        when the process is forked) rather than the constructor,
        this method should be extended to include initialization code
        to be run before the module simulation begins but after GPU
        initialization.
        
        """

        cuda.init()
        ctx = cuda.Device(self.device).make_context()
        atexit.register(ctx.pop)
        
    def run_step(self, in_non_gpu, in_spike_gpu, out_non_gpu, out_spike_gpu):
        """
        Run one step of the module simulation.

        Each step of the module's simulation consumes the data in
        `in_non_gpu` and `in_spike_gpu` and updates `out_non_gpu` and
        `out_spike_gpu`.
        
        Parameters
        ----------
        in_non_gpu : pycuda.gpuarray.GPUArray
            States of external non-spiking input neurons.
        in_spike_gpu : pycuda.gpuarray.GPUArray
            Indices of external spiking neurons that produced a
            spike at the previous simulation step.
        out_non_gpu : pycuda.gpuarray.GPUArray
            States of non-spiking output neurons.
        out_spike_gpu : pycuda.gpuarray.GPUArray
            Indices of spiking output neurons that produce a spike at
            the current simulation step.
           
        """

        raise NotImplementedError('You have to provide this method.')

    def __sync(self, in_non_gpu, in_spike_gpu, out_non_gpu, out_spike_gpu):
        """
        Propagate data to and from the module.

        Parameters
        ----------
        in_non_gpu : pycuda.gpuarray.GPUArray
            States of external non-spiking input neurons.
        in_spike_gpu : pycuda.gpuarray.GPUArray
            Indices of external spiking neurons that produced a
            spike at the previous simulation step.
        out_non_gpu : pycuda.gpuarray.GPUArray
            States of non-spiking output neurons.
        out_spike_gpu : pycuda.gpuarray.GPUArray
            Indices of spiking output neurons that produce a spike at
            the current simulation step.
        
        Returns
        -------
        done : bool
            If the received data is None, no data is propagated and
            the simulation is assumed to be over.
            
        Notes
        -----
        The non-spiking and spiking data arrays are transmitted
        together in a tuple. 

        The non-spiking data is an array if floats comprising the
        states of the external neurons. The spiking data is an array
        of integers listing the indices of the external neurons that
        have emitted a spike.
        
        """
        
        # Receive new input data:
        if is_poll_in(self.sock, self.poller):
            data = self.sock.recv_pyobj()
            if data == None:
                return False

            if type(data) != tuple or len(data) != 2:
                self.logger.info('received bad data')
                raise ValueError('received bad data')

            # The first entry in the received tuple is the non-spiking
            # data; the second is the spiking data:
            set_realloc(in_non_gpu, data[0])
            set_realloc(in_spike_gpu, data[1])

        # Send output data:
        self.sock.send_pyobj((out_non_gpu.get(), out_spike_gpu.get()))

        return True
    
    def run(self):
        """
        Body of process.
        
        """

        # Ignore Ctrl-C:
        orig_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Call initialization methods before simulation:
        self.init_gpu()
        self.init_zmq()

        # Main simulation loop:
        while True:

            # Propagate data between modules:
            done = self.__sync(
            # Run a step of the simulation:

            # Check whether the simulation should terminate:

        # Restore SIGINT signal handler before exiting:
        signal.signal(signal.SIGINT, orig_handler)
        self.logger.info('done')
        
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
