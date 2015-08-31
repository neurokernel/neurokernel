#!/usr/bin/env python

from multiprocessing import Queue
from unittest import main, TestCase

import numpy as np

from neurokernel.plsel import Selector, SelectorMethods
from neurokernel.pattern import Pattern
from neurokernel.core import Manager, Module
from neurokernel.tools.comm import get_random_port
from neurokernel.tools.logging import setup_logger

class TestModule(Module):
    def __init__(self, sel,
                 sel_in_gpot, sel_in_spike,
                 sel_out_gpot, sel_out_spike,
                 data_gpot, data_spike,
                 port_data, port_ctrl, port_time,
                 id):
        super(TestModule, self).__init__(sel,
                                       Selector.add(sel_in_gpot, sel_out_gpot),
                                       Selector.add(sel_in_spike, sel_out_spike),
                                       data_gpot, data_spike,
                                       ['interface', 'io', 'type'],
                                       port_data, port_ctrl, port_time,
                                       id, None, True, True)
        assert SelectorMethods.is_in(sel_in_gpot, sel)
        assert SelectorMethods.is_in(sel_out_gpot, sel)
        assert SelectorMethods.are_disjoint(sel_in_gpot, sel_out_gpot)
        assert SelectorMethods.is_in(sel_in_spike, sel)
        assert SelectorMethods.is_in(sel_out_spike, sel)
        assert SelectorMethods.are_disjoint(sel_in_spike, sel_out_spike)

        self.interface[sel_in_gpot, 'io', 'type'] = ['in', 'gpot']
        self.interface[sel_out_gpot, 'io', 'type'] = ['out', 'gpot']
        self.interface[sel_in_spike, 'io', 'type'] = ['in', 'spike']
        self.interface[sel_out_spike, 'io', 'type'] = ['out', 'spike']

def run_test(m0_sel_in_gpot, m0_sel_in_spike,
             m0_sel_out_gpot, m0_sel_out_spike,
             m1_sel_in_gpot, m1_sel_in_spike,
             m1_sel_out_gpot, m1_sel_out_spike):

    # Create test module classes with a queue installed in the destination
    # module to check that data was correctly propagated:
    class TestModule0(TestModule):
        def __init__(self, *args, **kwargs):
            super(TestModule0, self).__init__(*args, **kwargs)
            self.q = Queue()

        def run_step(self):
            self.log_info('saving data to queue before run step')
            if self.steps > 0:
                self.q.put((self.pm['gpot'][self._out_port_dict['gpot']['m1']].copy(),
                            self.pm['spike'][self._out_port_dict['spike']['m1']].copy()))
            super(TestModule0, self).run_step()

    class TestModule1(TestModule):
        def __init__(self, *args, **kwargs):
            super(TestModule1, self).__init__(*args, **kwargs)
            self.q = Queue()

        def run_step(self):
            super(TestModule1, self).run_step()
            self.log_info('saving data to queue after run step')
            if self.steps > 0:
                self.q.put((self.pm['gpot'][self._in_port_dict['gpot']['m0']].copy(),
                            self.pm['spike'][self._in_port_dict['spike']['m0']].copy()))

    m0_sel_gpot = m0_sel_in_gpot+m0_sel_out_gpot
    m0_sel_spike = m0_sel_in_spike+m0_sel_out_spike
    m0_sel = m0_sel_in_gpot+m0_sel_in_spike+m0_sel_out_gpot+m0_sel_out_spike
    m0_data_gpot = np.ones(len(m0_sel_gpot), np.double)
    m0_data_spike = np.ones(len(m0_sel_spike), np.int32)

    m1_sel_gpot = m1_sel_in_gpot+m1_sel_out_gpot
    m1_sel_spike = m1_sel_in_spike+m1_sel_out_spike
    m1_sel = m1_sel_in_gpot+m1_sel_in_spike+m1_sel_out_gpot+m1_sel_out_spike
    m1_data_gpot = np.zeros(len(m1_sel_gpot), np.double)
    m1_data_spike = np.zeros(len(m1_sel_spike), np.int32)

    # Instantiate manager and broker:
    man = Manager(get_random_port(), get_random_port(), get_random_port())
    man.add_brok()

    # Add modules:
    m0 = TestModule0(m0_sel, m0_sel_in_gpot, m0_sel_in_spike,
                     m0_sel_out_gpot, m0_sel_out_spike, 
                     m0_data_gpot, m0_data_spike,
                     man.port_data, man.port_ctrl, man.port_time,
                     id='m0')
    man.add_mod(m0)
    m1 = TestModule1(m1_sel, m1_sel_in_gpot, m1_sel_in_spike,
                     m1_sel_out_gpot, m1_sel_out_spike,
                     m1_data_gpot, m1_data_spike,
                     man.port_data, man.port_ctrl, man.port_time,
                     id='m1')
    man.add_mod(m1)

    # Connect the modules:
    pat = Pattern(m0_sel, m1_sel)
    pat.interface[m0_sel_in_gpot] = [0, 'in', 'gpot']
    pat.interface[m0_sel_out_gpot] = [0, 'out', 'gpot']
    pat.interface[m0_sel_in_spike] = [0, 'in', 'spike']
    pat.interface[m0_sel_out_spike] = [0, 'out', 'spike']

    pat.interface[m1_sel_in_gpot] = [1, 'in', 'gpot']
    pat.interface[m1_sel_out_gpot] = [1, 'out', 'gpot']
    pat.interface[m1_sel_in_spike] = [1, 'in', 'spike']
    pat.interface[m1_sel_out_spike] = [1, 'out', 'spike']
    for sel_from, sel_to in zip(m0_sel_out_gpot,
                                m1_sel_in_gpot):
        if not (sel_from == ((),) or sel_to == ((),)):
            pat[sel_from, sel_to] = 1
    for sel_from, sel_to in zip(m0_sel_out_spike,
                                m1_sel_in_spike):
        if not (sel_from == ((),) or sel_to == ((),)):
            pat[sel_from, sel_to] = 1

    man.connect(m0, m1, pat, 0, 1)

    # Execute exactly two steps; m0 transmits data during the first step, which
    # should be received by m1 during the second step:
    man.start(steps=2)
    man.stop()

    # Forcibly terminate all processes that are still alive:
    if m0.is_alive():
        m0.terminate()
    if m1.is_alive():
        m1.terminate()
    for b in man.brokers.values():
        if b.is_alive():
            b.terminate()

    # Check that data was propagated correctly:
    m0_data_gpot_after, m0_data_spike_after = m0.q.get()
    m1_data_gpot_after, m1_data_spike_after = m1.q.get()
    assert all(m0_data_gpot_after == m1_data_gpot_after)
    assert all(m0_data_spike_after == m1_data_spike_after)

class test_transmission(TestCase):
    def test_trans_gpot(self):
        m0_sel_in_gpot = Selector('')
        m0_sel_in_spike = Selector('')
        m0_sel_out_gpot = Selector('/m0/out/gpot[0:5]')
        m0_sel_out_spike = Selector('')
        m1_sel_in_gpot = Selector('/m1/in/gpot[0:5]')
        m1_sel_in_spike = Selector('')
        m1_sel_out_gpot = Selector('')
        m1_sel_out_spike = Selector('')

        run_test(m0_sel_in_gpot, m0_sel_in_spike, m0_sel_out_gpot, m0_sel_out_spike,
                 m1_sel_in_gpot, m1_sel_in_spike, m1_sel_out_gpot, m1_sel_out_spike)

    def test_trans_spike(self):
        m0_sel_in_gpot = Selector('')
        m0_sel_in_spike = Selector('')
        m0_sel_out_gpot = Selector('')
        m0_sel_out_spike = Selector('/m0/out/spike[0:5]')
        m1_sel_in_gpot = Selector('')
        m1_sel_in_spike = Selector('/m1/in/spike[0:5]')
        m1_sel_out_gpot = Selector('')
        m1_sel_out_spike = Selector('')

        run_test(m0_sel_in_gpot, m0_sel_in_spike, m0_sel_out_gpot, m0_sel_out_spike,
                 m1_sel_in_gpot, m1_sel_in_spike, m1_sel_out_gpot, m1_sel_out_spike)

    def test_trans_both(self):
        m0_sel_in_gpot = Selector('')
        m0_sel_in_spike = Selector('')
        m0_sel_out_gpot = Selector('/m0/out/gpot[0:5]')
        m0_sel_out_spike = Selector('/m0/out/spike[0:5]')
        m1_sel_in_gpot = Selector('/m1/in/gpot[0:5]')
        m1_sel_in_spike = Selector('/m1/in/spike[0:5]')
        m1_sel_out_gpot = Selector('')
        m1_sel_out_spike = Selector('')

        run_test(m0_sel_in_gpot, m0_sel_in_spike, m0_sel_out_gpot, m0_sel_out_spike,
                 m1_sel_in_gpot, m1_sel_in_spike, m1_sel_out_gpot, m1_sel_out_spike)


if __name__ == '__main__':
    main()
