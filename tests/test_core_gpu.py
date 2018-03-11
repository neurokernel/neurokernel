#!/usr/bin/env python

import neurokernel.mpi_relaunch

import cPickle as pickle
import os
import tempfile

from mpi4py import MPI
import numpy as np
import pycuda.gpuarray as gpuarray

from neurokernel.pattern import Pattern
from neurokernel.plsel import Selector, SelectorMethods
from neurokernel.core_gpu import Module, Manager, CTRL_TAG, GPOT_TAG, SPIKE_TAG
import neurokernel.mpi as mpi

class MyModule1(Module):
    """
    Module that emits data.
    """

    def __init__(self, sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns=['interface', 'io', 'type'],
                 ctrl_tag=CTRL_TAG, gpot_tag=GPOT_TAG, spike_tag=SPIKE_TAG,
                 id=None, device=None,
                 routing_table=None, rank_to_id=None,
                 debug=False, time_sync=False, out_spike_data=None):
        super(MyModule1, self).__init__(sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns,
                 ctrl_tag, gpot_tag, spike_tag,
                 id, device,
                 routing_table, rank_to_id,
                 debug, time_sync)
        self.out_spike_data = out_spike_data

    def run_step(self):
        super(MyModule1, self).run_step()

        # Emit data by setting elements in port map data array corresponding to
        # output ports:
        if self.out_spike_data:
            out_spike_data_gpu = gpuarray.to_gpu(np.asarray(self.out_spike_data,
                                                        self.data['spike'].dtype))
            self.pm['spike'][self.out_spike_ports] = out_spike_data_gpu
            self.log_info('output spike port data: '+str(out_spike_data_gpu))

class MyModule2(Module):
    """
    Module that expects data.
    """

    def __init__(self, sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns=['interface', 'io', 'type'],
                 ctrl_tag=CTRL_TAG, gpot_tag=GPOT_TAG, spike_tag=SPIKE_TAG,
                 id=None, device=None,
                 routing_table=None, rank_to_id=None,
                 debug=False, time_sync=False, out_file_name=None):
        super(MyModule2, self).__init__(sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns,
                 ctrl_tag, gpot_tag, spike_tag,
                 id, device,
                 routing_table, rank_to_id,
                 debug, time_sync)
        self.out_file_name = out_file_name
            
    out_buf = []

    def run_step(self):
        super(MyModule2, self).run_step()

        # Save received data by recording elements in port map data array
        # corresponding to input ports:
        in_spike_data = self.pm['spike'][self.in_spike_ports]
        self.log_info('input spike port data: '+str(in_spike_data))
        self.out_buf.append(in_spike_data)

    def post_run(self):
        if self.out_file_name:
            with open(self.out_file_name, 'w') as f:
                pickle.dump(self.out_buf[1], f)
        super(MyModule2, self).post_run()

def make_sels(sel_in_gpot, sel_out_gpot, sel_in_spike, sel_out_spike):
    sel_in_gpot = Selector(sel_in_gpot)
    sel_out_gpot = Selector(sel_out_gpot)
    sel_in_spike = Selector(sel_in_spike)
    sel_out_spike = Selector(sel_out_spike)

    sel = sel_in_gpot+sel_out_gpot+sel_in_spike+sel_out_spike
    sel_in = sel_in_gpot+sel_in_spike
    sel_out = sel_out_gpot+sel_out_spike
    sel_gpot = sel_in_gpot+sel_out_gpot
    sel_spike = sel_in_spike+sel_out_spike
    
    return sel, sel_in, sel_out, sel_gpot, sel_spike

from unittest import main, TestCase

debug = False

class test_core_gpu(TestCase):
    def setUp(self):
        self.man = Manager()

    def test_transmit_spikes_one_to_one(self):
        m1_sel_in_gpot = Selector('')
        m1_sel_out_gpot = Selector('')
        m1_sel_in_spike = Selector('')
        m1_sel_out_spike = Selector('/m1/out/spike[0:4]')
        m1_sel, m1_sel_in, m1_sel_out, m1_sel_gpot, m1_sel_spike = \
            make_sels(m1_sel_in_gpot, m1_sel_out_gpot, m1_sel_in_spike, m1_sel_out_spike)
        N1_gpot = SelectorMethods.count_ports(m1_sel_gpot)
        N1_spike = SelectorMethods.count_ports(m1_sel_spike)

        m2_sel_in_gpot = Selector('')
        m2_sel_out_gpot = Selector('')
        m2_sel_in_spike = Selector('/m2/in/spike[0:4]')
        m2_sel_out_spike = Selector('')
        m2_sel, m2_sel_in, m2_sel_out, m2_sel_gpot, m2_sel_spike = \
            make_sels(m2_sel_in_gpot, m2_sel_out_gpot, m2_sel_in_spike, m2_sel_out_spike)
        N2_gpot = SelectorMethods.count_ports(m2_sel_gpot)
        N2_spike = SelectorMethods.count_ports(m2_sel_spike)

        m1_id = 'm1'
        self.man.add(MyModule1, m1_id,
                     m1_sel, m1_sel_in, m1_sel_out,
                     m1_sel_gpot, m1_sel_spike,
                     np.zeros(N1_gpot, dtype=np.double),
                     np.zeros(N1_spike, dtype=int),
                     device=0, debug=debug, out_spike_data=[0, 0, 1, 1])

        f, out_file_name = tempfile.mkstemp()
        os.close(f)

        m2_id = 'm2'
        self.man.add(MyModule2, m2_id,
                     m2_sel, m2_sel_in, m2_sel_out,
                     m2_sel_gpot, m2_sel_spike,
                     np.zeros(N2_gpot, dtype=np.double),
                     np.zeros(N2_spike, dtype=int),
                     device=1, debug=debug, out_file_name=out_file_name)

        pat12 = Pattern(m1_sel, m2_sel)
        pat12.interface[m1_sel_out_gpot] = [0, 'in', 'gpot']
        pat12.interface[m1_sel_in_gpot] = [0, 'out', 'gpot']
        pat12.interface[m1_sel_out_spike] = [0, 'in', 'spike']
        pat12.interface[m1_sel_in_spike] = [0, 'out', 'spike']
        pat12.interface[m2_sel_in_gpot] = [1, 'out', 'gpot']
        pat12.interface[m2_sel_out_gpot] = [1, 'in', 'gpot']
        pat12.interface[m2_sel_in_spike] = [1, 'out', 'spike']
        pat12.interface[m2_sel_out_spike] = [1, 'in', 'spike']
        pat12['/m1/out/spike[0]', '/m2/in/spike[0]'] = 1
        pat12['/m1/out/spike[1]', '/m2/in/spike[1]'] = 1
        pat12['/m1/out/spike[2]', '/m2/in/spike[2]'] = 1
        pat12['/m1/out/spike[3]', '/m2/in/spike[3]'] = 1
        self.man.connect(m1_id, m2_id, pat12, 0, 1)

        # Run emulation for 2 steps:
        self.man.spawn()
        self.man.start(2)
        self.man.wait()

        # Get output of m2:
        with open(out_file_name, 'r') as f:
            output = pickle.load(f)

        os.remove(out_file_name)
        self.assertSequenceEqual(list(output), [0, 0, 1, 1])

    def test_transmit_spikes_one_to_many(self):
        m1_sel_in_gpot = Selector('')
        m1_sel_out_gpot = Selector('')
        m1_sel_in_spike = Selector('')
        m1_sel_out_spike = Selector('/m1/out/spike[0:4]')
        m1_sel, m1_sel_in, m1_sel_out, m1_sel_gpot, m1_sel_spike = \
            make_sels(m1_sel_in_gpot, m1_sel_out_gpot, m1_sel_in_spike, m1_sel_out_spike)
        N1_gpot = SelectorMethods.count_ports(m1_sel_gpot)
        N1_spike = SelectorMethods.count_ports(m1_sel_spike)

        m2_sel_in_gpot = Selector('')
        m2_sel_out_gpot = Selector('')
        m2_sel_in_spike = Selector('/m2/in/spike[0:4]')
        m2_sel_out_spike = Selector('')
        m2_sel, m2_sel_in, m2_sel_out, m2_sel_gpot, m2_sel_spike = \
            make_sels(m2_sel_in_gpot, m2_sel_out_gpot, m2_sel_in_spike, m2_sel_out_spike)
        N2_gpot = SelectorMethods.count_ports(m2_sel_gpot)
        N2_spike = SelectorMethods.count_ports(m2_sel_spike)

        m1_id = 'm1'
        self.man.add(MyModule1, m1_id,
                     m1_sel, m1_sel_in, m1_sel_out,
                     m1_sel_gpot, m1_sel_spike,
                     np.zeros(N1_gpot, dtype=np.double),
                     np.zeros(N1_spike, dtype=int),
                     device=0, debug=debug, out_spike_data=[1, 0, 0, 0])

        f, out_file_name = tempfile.mkstemp()
        os.close(f)

        m2_id = 'm2'
        self.man.add(MyModule2, m2_id,
                     m2_sel, m2_sel_in, m2_sel_out,
                     m2_sel_gpot, m2_sel_spike,
                     np.zeros(N2_gpot, dtype=np.double),
                     np.zeros(N2_spike, dtype=int),
                     device=1, debug=debug, out_file_name=out_file_name)

        pat12 = Pattern(m1_sel, m2_sel)
        pat12.interface[m1_sel_out_gpot] = [0, 'in', 'gpot']
        pat12.interface[m1_sel_in_gpot] = [0, 'out', 'gpot']
        pat12.interface[m1_sel_out_spike] = [0, 'in', 'spike']
        pat12.interface[m1_sel_in_spike] = [0, 'out', 'spike']
        pat12.interface[m2_sel_in_gpot] = [1, 'out', 'gpot']
        pat12.interface[m2_sel_out_gpot] = [1, 'in', 'gpot']
        pat12.interface[m2_sel_in_spike] = [1, 'out', 'spike']
        pat12.interface[m2_sel_out_spike] = [1, 'in', 'spike']
        pat12['/m1/out/spike[0]', '/m2/in/spike[0]'] = 1
        pat12['/m1/out/spike[0]', '/m2/in/spike[1]'] = 1
        pat12['/m1/out/spike[0]', '/m2/in/spike[2]'] = 1
        pat12['/m1/out/spike[0]', '/m2/in/spike[3]'] = 1
        self.man.connect(m1_id, m2_id, pat12, 0, 1)

        # Run emulation for 2 steps:
        self.man.spawn()
        self.man.start(2)
        self.man.wait()

        # Get output of m2:
        with open(out_file_name, 'r') as f:
            output = pickle.load(f)

        os.remove(out_file_name)
        self.assertSequenceEqual(list(output), [1, 1, 1, 1])

if __name__ == '__main__':
    logger = mpi.setup_logger(screen=False,
                              mpi_comm=MPI.COMM_WORLD, multiline=True)
    main()
