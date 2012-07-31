import sys
import random as rd
import scipy as sp
import numpy.random as np_rd
from time import gmtime, strftime
from collections import namedtuple as Pulse
import pdb
import itertools

import pycuda.gpuarray as garray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
import atexit

from neurokernel.tools import parray
from neurokernel.Module import Module
from neurokernel.tools.misc_utils import rand_bin_matrix

class MockSystem(Module):
    """
    Neural network class. This code, by now, is provided by the user. In this
    example, this code is the lamina version implemented by Nikul and Yiyin.
    """
    def __init__(self, N_spk, N_gpot, N_synapses,
                 dt, N_gpot_proj, N_spk_proj, device, N_inputs):
        """
        Synaptic connectivity between modules.

        Attributes
        ----------
        manager : neurokernel.Manager
            Synaptic connectivity.
        num_spk : int
            Number of spiking neurons.
        num_gpot : int
            Number of graded-potential neurons.
        num_synapses : int
            Number of synapses for spiking and graded-potential neurons: Half
            each.
        dt : double
            Duration of each simulation's step.
        num_gpot_proj : int
            Number of output graded-potential neurons.
        num_spk_proj : int
            Number of output spiking neurons.
        device : int
            GPU device number.
        num_inputs : int
            Number of module's inputs.

        """

        np_rd.seed(0)

        Module.__init__(self, dt, N_inputs, N_gpot_proj, N_spk_proj, device)

        self.num_gpot = N_gpot
        self.num_spk = N_spk
        self.num_synapses = N_synapses

    def init_gpu(self):

        # If there is graded-potential neurons to be processed
        if (self.num_gpot > 0):
            # In order to understand pre_neuron, post_neuron and dendrites
            # it's necessary notice that the process is over the synapses
            # instead of neurons. So, in fact there is no neurons, but
            # connection between neurons. Number of dendrites per neuron.
            # A dendrite is a neuron's
            gpot_synapses = self.num_synapses / 2

            pre_neuron = np_rd.random_integers(0, self.num_gpot,
                                    size = (gpot_synapses,)).astype(np.int32)

            post_neuron = np.sort(np_rd.random_integers(0, self.num_gpot,
                                    size = (gpot_synapses,)).astype(np.int32))

            self.num_dendrites = sp.bincount(post_neuron)

            # Parameters of the model: threshold, slope, saturation, Vs
            # and phy.
            # Shape: (num_synapses,)
            thres = np.asarray([rd.gauss(-.5, .01) for x in \
                                np.zeros([gpot_synapses])], dtype = np.float64)
            slope = np.asarray([rd.gauss(-.5, .1) for x in \
                                np.zeros([gpot_synapses])], dtype = np.float64)
            saturation = np.asarray([rd.gauss(.1, .01) for x in \
                                np.zeros([gpot_synapses])], dtype = np.float64)
            power = np.ones([gpot_synapses], dtype = np.float64)
            reverse = np.asarray([rd.gauss(-.4, .1) for x in \
                                np.zeros([gpot_synapses])], dtype = np.float64)

            # Parameters of alpha function. Shape: (num_synapses,)
            delay = np.ones([gpot_synapses], dtype = np.float64)

            # Initial condition at resting potential. Shape: (num_gpot,)
            V = np.asarray([rd.gauss(-.51, .01) for x in \
                            np.zeros([self.num_gpot])], dtype = np.float64)
            n = np.asarray([rd.gauss(.3, .05) for x in \
                            np.zeros([self.num_gpot])], dtype = np.float64)

            self.delay_steps = int(round(max(delay) * 1e-3 / self.dt))

            self.buffer = CircularArray(self.num_gpot, self.delay_steps, V)

            self.gpot_neu = MorrisLecar(self.num_gpot,
                                       self.dt, self.num_dendrites, V, n,
                                       self.N_inputs)
            self.gpot_syn = VectorSynapse(gpot_synapses, pre_neuron,
                                          post_neuron, thres, slope, power,
                                          saturation, delay, reverse, self.dt)
        if (self.num_spk > 0):
            self.olfnet = IAFNet(70, 100)
            self.olfnet.gpu_step_prepare()

    def run_step(self, in_list = None, proj_list = None):

        self.gpot_neu.I_pre.fill(0)
        self.gpot_neu.update_I_pre_input(in_list[0])
        self.gpot_neu.read_synapse(self.gpot_syn.conductance,
                                  self.gpot_syn.V_rev)
        self.gpot_neu.eval(self.buffer)
        self.gpot_syn.compute_synapse(self.buffer)
        cuda.memcpy_dtoh(proj_list[0], self.gpot_neu.V.gpudata)
        self.buffer.step()
        self.olfnet.run_step()
        self.olfnet.gpu_spk_list.get()

class CircularArray:
    """
    GP neurons.

    """
    def __init__(self, num_neurons, delay_steps, rest):
        self.dtype = np.double
        self.num_neurons = num_neurons
        self.delay_steps = delay_steps

        self.buffer = parray.empty((delay_steps, num_neurons), np.double)

        d_rest = garray.to_gpu(rest)
        self.current = 0

        #initializing V buffer
        for i in range(delay_steps):
            cuda.memcpy_dtod(int(self.buffer.gpudata) + self.buffer.ld * i * \
                             self.buffer.dtype.itemsize, d_rest.gpudata,
                             d_rest.nbytes)

    def step(self):
        self.current += 1
        if self.current >= self.delay_steps:
            self.current = 0

class MorrisLecar:
    """
    GP neurons.

    """
    def __init__(self, num_neurons, dt, num_dendrite, V, n,
                 num_inputs):
        """
        Set Morris Lecar neurons in the network.

        Parameters
        ----------
        N : int
            Number of neurons to be added.
        """
        self.dtype = np.double
        self.num_neurons = num_neurons
        self.dt = dt
        self.steps = max(int(round(dt / 1e-5)), 1)

        self.ddt = dt / self.steps

        self.V = garray.to_gpu(V)
        self.n = garray.to_gpu(n)
        self.num_types = 15
        self.num_cart = 768

        self.I_pre = garray.zeros(self.num_neurons, np.double)

        self.h_V = cuda.pagelocked_empty((self.num_types, self.num_cart),
                                         np.double)

        self.cum_num_dendrite = garray.to_gpu(np.concatenate((np.asarray([0, ],
                                                dtype = np.int32),
                                                np.cumsum(num_dendrite,
                                                          dtype = np.int32))))
        self.num_dendrite = garray.to_gpu(num_dendrite)
        self.num_input = num_inputs

        self.update = self.get_euler_kernel()
        self.get_input = self.get_input_func()

    def update_I_pre_input(self, I_ext):
        cuda.memcpy_dtod(int(self.I_pre.gpudata), I_ext,
                         self.num_input * self.I_pre.dtype.itemsize)

    def read_synapse(self, conductance, V_rev, st = None):
        self.get_input.prepared_async_call(self.grid_get_input,
                                           self.block_get_input, st,
                                           conductance.gpudata,
                                           self.cum_num_dendrite.gpudata,
                                           self.num_dendrite.gpudata,
                                           self.I_pre.gpudata, self.V.gpudata,
                                           V_rev.gpudata)

    def eval(self, buffer, st = None):
        self.update.prepared_async_call(self.update_grid, self.update_block,
                                        st, self.V.gpudata, self.n.gpudata,
                                        int(buffer.buffer.gpudata) + \
                                        buffer.current * buffer.buffer.ld * \
                                        buffer.buffer.dtype.itemsize,
                                        self.num_neurons, self.I_pre.gpudata,
                                        self.ddt * 1000, self.steps)

    def get_euler_kernel(self):
        template = open('neurokernel/mock_sys/cuda_code/euler_kernel.cu', 'r')

        dtype = self.dtype
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128, 1, 1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template.read() % {"type": dtype_to_ctype(dtype),
                                              "ntype": self.num_types,
                                              "nneu": self.update_block[0]},
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("hhn_euler_multiple")

        func.prepare([np.intp, np.intp, np.intp, np.int32, np.intp, scalartype,
                      np.int32])

        return func

    def get_input_func(self):
        template = open('neurokernel/mock_sys/cuda_code/input_func.cu', 'r')

        mod = SourceModule(template.read() % {"num_neurons": self.num_neurons},
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.block_get_input = (32, 32, 1)
        self.grid_get_input = ((self.num_neurons - 1) / 32 + 1, 1)

        return func

class VectorSynapse:
    """
    GP neurons.

    """
    def __init__(self, num_synapse, pre_neuron, post_neuron, syn_thres,
                 syn_slope, syn_power, syn_saturation, syn_delay, V_rev, dt):

        self.dt = dt
        self.num_synapse = num_synapse
        self.pre_neuron = garray.to_gpu(pre_neuron)
        #self.post_neuron = garray.to_gpu(post_neuron)

        self.threshold = garray.to_gpu(syn_thres)
        self.slope = garray.to_gpu(syn_slope)
        self.power = garray.to_gpu(syn_power)
        self.saturation = garray.to_gpu(syn_saturation)
        self.delay = garray.to_gpu(
                            np.round(syn_delay * 1e-3 / dt).astype(np.int32))
        self.conductance = garray.zeros(self.num_synapse, np.double)

        self.V_rev = garray.to_gpu(V_rev)

        self.update_terminal_synapse = self.get_update_terminal_synapse_func()
        self.mem_tmp = garray.empty(self.num_synapse, np.double)

    def compute_synapse(self, buffer, st = None):
        self.update_terminal_synapse.prepared_async_call(
                                                    self.grid_terminal_synapse,
                                                    self.block_terminal_synapse,
                                                    st, buffer.buffer.gpudata,
                                                    buffer.buffer.ld,
                                                    buffer.current,
                                                    buffer.delay_steps,
                                                    self.pre_neuron.gpudata,
                                                    self.conductance.gpudata,
                                                    self.threshold.gpudata,
                                                    self.slope.gpudata,
                                                    self.power.gpudata,
                                                    self.saturation.gpudata,
                                                    self.delay.gpudata,
                                                    self.mem_tmp.gpudata)

    def get_update_terminal_synapse_func(self):
        template = open('neurokernel/mock_sys/cuda_code/terminal_synapse.cu',
                        'r')

        mod = SourceModule(template.read() % {"n_synapse": self.num_synapse},
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("update_terminal_synapse")
        func.prepare([np.intp, np.int32, np.int32, np.int32, np.intp, np.intp,
                      np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.block_terminal_synapse = (256, 1, 1)
        self.grid_terminal_synapse = (min(6 * \
                              cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                              (self.num_synapse - 1) / 256 + 1), 1)

        return func

class AlphaSyn:
    def __init__(self, neu_list, neu_coef, gmax, tau, sign = 1):
        self.neu_list = neu_list
        self.neu_coef = neu_coef
        self.taur = 1.0 / tau
        self.gmax = gmax
        self.gvec = np.array([0., 0., 0.]) #[g(t) g'(t) g"(t)] 
        self.sign = sign # -1:inhibitory; 1:excitatory

    def _get_g(self):
        return self.gvec[0] * self.gmax

    g = property(_get_g)

class IAFNeu:
    def __init__(self, V0, Vr, Vt, tau, R, syn_list):
        self.V = V0
        self.Vr = Vr
        self.Vt = Vt

        self.tau = tau
        self.R = R

        self.syn_list = list(syn_list)
        self.I = 0
        self.spk = False

class IAFNet:
    def __init__(self, num_neurons, num_syn):
        self.neu_list = []
        self.neu_name = {}
        self.syn_list = []
        self.syn_name = {}
        self.dt = 1e-5
        self.dur = 1.0

        self.spk_list = []
        self.neu_num = num_neurons
        self.syn_num = num_syn

        self.max_thread = 512

        self.readNeuron()
        self.readSynapse()

    def readNeuron(self):
        for i in xrange(self.neu_num):
            self.neu_name[ str(i) ] = len(self.neu_name)
            self.neu_list.append(IAFNeu(-0.05, -0.07, -0.035, 0.4, 0.9, []))

    def neuAppendSyn(self, post_neu, syn_idx = -1):
        if syn_idx == -1: syn_idx = len(self.syn_list)
        self.neu_list[ self.neu_name[post_neu] ].syn_list.append(syn_idx)

    def readSynapse(self):
        x = rand_bin_matrix((self.neu_num, self.neu_num), self.syn_num)
        c = [a for a in itertools.product(range(x.shape[0]),
                                          range(x.shape[1])) if x[a[0], a[1]]]
        for i in xrange(self.syn_num):
            pre_neu, post_neu = c[i]

            name = str(pre_neu) + '-' + str(post_neu)
            self.syn_name[ name ] = len(self.syn_name)
            self.neuAppendSyn(str(post_neu))
            self.syn_list.append(AlphaSyn([self.neu_name[str(pre_neu)]],
                                          [rd.gauss(435, 80)],
                                          float(np_rd.randint(1, 3)) / 2,
                                          rd.gauss(.057, .004),
                                          np_rd.randint(0, 2) - 1))

    def readPreSyn(self, f, presyn_num):
        pdb.set_trace()
        for i in xrange(presyn_num):
            lineInFile = myreadline(f)
            ln_neu, pre_neu, post_neu, coef = lineInFile.split()
            pdb.set_trace()
            syn_name = pre_neu + '-' + post_neu
            if self.neu_name.has_key(ln_neu) == False:
                raise IOError('No such Local Neuron: ' + ln_neu)
            if self.syn_name.has_key(syn_name) == False:
                raise IOError('No such Synapse: ' + syn_name)
            syn_idx = self.syn_name[ syn_name ]
            self.syn_list[syn_idx].neu_list.append(self.neu_name[ ln_neu ])
            pdb.set_trace()
            self.syn_list[syn_idx].neu_coef.append(float(coef))

    def readOneLineCurrent(self, line):
        name, pline = line.split(None, 1)
        if self.neu_name.has_key(name) == False:
            raise IOError("In: " + line + "\nNo such Neuron: " + name)
        while True:
            seg = pline.split(None, 3)
            if len(seg) < 3:
                raise IOError("In: " + line + "\n"\
                         "Pulse contains beginning, end, and value: " + pline)
            if self.curr_list.has_key(name) == False:
                self.curr_list[name] = []
            tmp = Pulse(float(seg[0]), float(seg[1]), float(seg[2]))
            if tmp.start >= tmp.end:
                raise IOError("In: " + line + "\n"\
                         "Pulse Beginning should be less than Pulse End: " \
                         + repr(tmp))
            self.curr_list[name].append(tmp)
            if len(seg) == 3: break
            pline = seg[3]

    def readCurrentFromFile(self, filename):
        f = open(filename, 'r')
        self.curr_list = {}
        while True:
            s = myreadline(f)
            if s == '': break
            self.readOneLineCurrent(s)

    def genCurrent(self, I_ext = np.zeros((0, 0))):
        self.curr_list = {}
        self.neu_I_ext_map = -1 * np.ones(self.neu_num, dtype = np.int32)
        if I_ext.size > 0:
            self.I_ext = I_ext.astype(np.float64)
            self.neu_cur_map[:I_ext.shape[0]] = range(I_ext.shape[0])
            return
        # Find number of neuron who has external current
        max_pulse_end = 0
        neu_w_curr = []
        for name, pulse_list in self.curr_list.items():
            neu_w_curr.append(self.neu_name[ name ])
            for pulse in pulse_list:
                if max_pulse_end < pulse.end:
                    max_pulse_end = pulse.end
        neu_w_curr.sort()
        self.I_ext = np.zeros((int(max_pulse_end / self.dt), len(neu_w_curr)))
        # 
        for name, pulse_list in self.curr_list.items():
            neu_idx = self.neu_name[ name ]
            cur_idx = neu_w_curr.index(neu_idx)
            self.neu_I_ext_map[ neu_idx ] = cur_idx
            for pulse in pulse_list:
                self.I_ext[int(pulse.start / self.dt):\
                           int(pulse.end / self.dt), cur_idx] = pulse.value
        t = np.arange(int(max_pulse_end / self.dt)) * self.dt

    def basic_prepare(self, dt = 0., dur = 0., I_ext = np.zeros((0, 0))):
        if self.neu_num == 0:
            raise IOError("Can't run simulation without any neuron...")
        self.dt = self.dt if dt == 0. else dt
        self.dur = self.dur if dur == 0. else dur
        if self.dt <= 0.:
            raise IOError("dt should be declared or greater than zero.")
        if self.dur <= 0.:
            raise IOError("Duration should be declared or greater than zero.")
        self.Nt = int(self.dur / self.dt)
        self.spk_list = np.zeros((self.Nt, self.neu_num), np.int32)
        self.genCurrent(I_ext)

    def list_notempty(self, arr):
        # Return dummy array if the input is empty. The empty array will 
        # cause exception when one tries to use driver.In()
        return arr if arr.size > 0 else np.zeros(1)

    def gpu_prepare(self, dt = 0., dur = 0., I_ext = np.empty((0, 0))):
        self.basic_prepare(dt, dur, I_ext)
        # Merge Neuron data
        gpu_neu_list = np.zeros(self.neu_num, dtype = ('f8,f8,f8,f8,f8,i4,i4'))
        offset, agg_syn = 0, []
        for i in xrange(self.neu_num):
            n = self.neu_list[i]
            gpu_neu_list[i] = (n.V, n.Vr, n.Vt, n.tau,
                                n.R, len(n.syn_list), offset)
            offset += len(n.syn_list)
            agg_syn.extend(n.syn_list)
        gpu_neu_syn_list = self.list_notempty(np.array(agg_syn,
                                                       dtype = np.int32))

        # Merge Synapse data
        gpu_syn_list = self.list_notempty(np.zeros(self.syn_num, dtype =
                                           ('f8,f8,f8,f8,f8,f8,f8,i4,i4')))
        offset, agg_neu, agg_coe = 0, [], []
        for i in xrange(self.syn_num):
            s = self.syn_list[i]
            gpu_syn_list[i] = (s.g, np.float64(0.0), np.float64(0.0),
                               np.float64(0.0), s.gmax, s.taur,
                               s.sign, len(s.neu_list), offset)
            offset += len(s.neu_list)
            agg_neu.extend(s.neu_list)
            agg_coe.extend(s.neu_coef)
        gpu_syn_neu_list = self.list_notempty(np.array(zip(agg_neu, agg_coe),
                                                       dtype = ('i8,f8')))

        # Determine Bloack and Grid size
        num = max(self.neu_num, self.syn_num)
        if num % self.max_thread == 0:
            gridx = (num / self.max_thread)
        else:
            gridx = 1 + num / self.max_thread
        return gridx, gpu_neu_list, gpu_neu_syn_list, gpu_syn_list, \
               gpu_syn_neu_list

    def gpu_run(self, dt = 0., dur = 0., I_ext = np.empty((0, 0))):
        gridx, neu_list, neu_syn_list, syn_list, syn_neu_list = \
            self.gpu_prepare(dt, dur)
        cuda_gpu_run(np.int32(self.Nt), np.double(self.dt),
                      np.int32(self.neu_num),
                      drv.In(neu_list), drv.In(neu_syn_list),
                      np.int32(self.syn_num),
                      drv.In(syn_list), drv.In(syn_neu_list),
                      drv.Out(self.spk_list),
                      drv.In(self.neu_I_ext_map.astype(np.int32)),
                      np.int32(self.I_ext.shape[1]),
                      np.int32(self.I_ext.shape[0]),
                      drv.In(self.list_notempty(
                                            self.I_ext.astype(np.float64))),
                      block = (self.max_thread, 1, 1), grid = (gridx, 1))

    def gpu_step_prepare(self):
        self.gridx, neu_list, neu_syn_list, syn_list, \
            syn_neu_list = self.gpu_prepare()
        self.gpu_neu_list = garray.to_gpu(neu_list)
        self.gpu_syn_list = garray.to_gpu(syn_list)
        self.gpu_neu_syn_list = garray.to_gpu(neu_syn_list)
        self.gpu_syn_neu_list = garray.to_gpu(syn_neu_list)
        self.gpu_neu_I_ext_map = garray.to_gpu(self.neu_I_ext_map)
        self.gpu_spk_list = garray.to_gpu(np.zeros(self.neu_num,
                                                     dtype = np.int32))
        self.gpu_I_list = garray.to_gpu(np.zeros(self.neu_num,
                                                   dtype = np.float64))

    def run_step(self):
        # The data from Interface will be concantenated to
        cuda_source = open('neurokernel/mock_sys/cuda_code/olf_gpu.cu', 'r')
        cuda_func = SourceModule(cuda_source.read(),
                                 options = ["--ptxas-options=-v"])
        cuda_gpu_run_dt = cuda_func.get_function("gpu_run_dt")
        cuda_gpu_run_dt(np.double(self.dt),
                         np.int32(self.neu_num), #number of neurons
                         np.int32(self.syn_num), #number of synapses
                         self.gpu_neu_list, #array of neuron status 
                         self.gpu_neu_syn_list,
                         self.gpu_syn_list, #array of synapse status
                         self.gpu_syn_neu_list, #array of pre-synaptic
                                                #inhibitory neuron
                         self.gpu_neu_I_ext_map,
                         self.gpu_I_list, #array of external current
                         self.gpu_spk_list, #output spikes
                         block = (self.max_thread, 1, 1), grid = (self.gridx, 1))

    def run(self):
        self.gpu_step_prepare()
        for i in xrange(self.Nt):
            self.run_step()
            self.spk_list[i, :] = self.gpu_spk_list.get()

def main(argv):
    try:
        num_spk = int(sys.argv[1])
        num_gpot = int(sys.argv[2])
        num_synapses = int(sys.argv[3])
        dt = np.double(sys.argv[4])
        num_gpot_proj = int(sys.argv[5])
        num_spk_proj = int(sys.argv[6])
        device = int(sys.argv[7])
        num_inputs = int(sys.argv[8])
    except IOError:
        print "Wrong #parameters. Exemple: 1000 1000 10000 1e-4 20 10 1 100"

    cuda.init()
    ctx = cuda.Device(device).make_context()
    atexit.register(ctx.pop)

    start = cuda.Event()
    end = cuda.Event()

    system = MockSystem(num_spk, num_gpot, num_synapses, dt, num_gpot_proj,
                        num_spk_proj, device, num_inputs)

    system.init_gpu()

    # External current
    I_ext = parray.to_gpu(np.ones([1 / system.dt, system.N_inputs]))
    out = np.empty((1 / system.dt, num_gpot_proj), np.double)

    start.record()
    for i in range(int(1 / system.dt)):
        temp = int(I_ext.gpudata) + I_ext.dtype.itemsize * I_ext.ld * i
        system.run_step([temp], [out[i, :]])

    end.record()
    end.synchronize()
    secs = start.time_till(end) * 1e-3
    print "Time: %fs" % secs

if __name__ == '__main__':
    main(sys.argv[1:])
