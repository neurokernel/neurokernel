import atexit, signal
import numpy as np
import multiprocessing as mp
import pycuda.driver as cuda

import twiggy

from neurokernel.tools import parray
from neurokernel.tools.comm_utils import is_poll_in
from neurokernel.tools.gpu_utils import set_realloc

class Module(mp.Process):
    """
    A module comprises one or more local processing units and it is the
    interface between those LPU and the manager.

    Parameters
    ----------
    manager : neurokernel.Manager
        Module manager that manages this module instance.
    dt : numpy.double
        Time resolution.
    device : int
        GPU device used by the module instance.

    See Also
    --------
    Manager.Manager

    """

    def __init__(self, dt, inputs, N_gpot_proj, N_spike_proj, device):

        super(Module, self).__init__()

        self.device = device

        # List of connection objects:
        self.conn_list = []

    def init_net(self):
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

    def run_step(self, in_gpot_list, in_spike_list, out_gpot_gpu, out_spike_gpu):
        """
        Run one step of the module simulation.

        Each step of the module's simulation consumes the data in the
        arrays listed in `in_gpot_list` and `in_spike_list` and updates
        `out_gpot_gpu` and `out_spike_gpu`.

        Parameters
        ----------
        in_gpot_list : list of pycuda.gpuarray.GPUArray
            States of external graded-potential input neurons.
        in_spike_list : list of pycuda.gpuarray.GPUArray
            Indices of external spiking neurons that produced a
            spike at the previous simulation step.
        out_gpot_gpu : pycuda.gpuarray.GPUArray
            States of non-spiking output neurons.
        out_spike_gpu : pycuda.gpuarray.GPUArray
            Indices of spiking output neurons that produce a spike at
            the current simulation step.

        """

        raise NotImplementedError('You have to provide this method.')

    def sync(self, in_gpot_list, in_spike_list, out_gpot_gpu, out_spike_gpu):
        """
        Propagate data to and from the module.

        Given GPUArrays instantiated to contain input and output data
        accessed by the module during a single step of simulation,
        update the input arrays by receiving data from source modules
        and send the data in the output arrays to destination modules.

        Parameters
        ----------
        in_gpot_gpu : pycuda.gpuarray.GPUArray
            States of external graded-potential input neurons.
        in_spike_gpu : pycuda.gpuarray.GPUArray
            Indices of external spiking neurons that produced a
            spike at the previous simulation step.
        out_gpot_gpu : pycuda.gpuarray.GPUArray
            States of graded-potential output neurons.
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
        The contents of the graded-potential and spiking data arrays
        are transmitted together in a tuple.

        The graded-potential data is an array of floats comprising the
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
        self.init_net()

        # Main simulation loop:
        while True:
            pass

            # Propagate data between modules:

            # Run a step of the simulation:

            # Check whether the simulation should terminate:

        # Restore SIGINT signal handler before exiting:
        signal.signal(signal.SIGINT, orig_handler)
        self.logger.info('done')
