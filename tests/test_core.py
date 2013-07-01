#!/usr/bin/env python

"""
Unit tests for neurokernel.core.
"""

from unittest import main, TestCase, TestSuite

import neurokernel.core as core

class test_IntervalIndex(TestCase):
    def test_one_interval(self):
        i = core.IntervalIndex([0, 5], 'a')
        assert i[1] == 1
        assert i['a', 1] == 1
        assert i['a', 1:3] == slice(1, 3, None)
        
    def test_two_intervals(self):
        i = core.IntervalIndex([0, 5, 10], ['a', 'b'])
        assert i[1] == 1
        assert i[6] == 1
        assert i['a', 1] == 1
        assert i['b', 1] == 6
        assert i['b', 1:3] == slice(6, 8, None)

class test_Connectivity(TestCase):
    def test_src_mask_no_conn(self):
        c = core.Connectivity(2, 2, 3, 3)
        all(c.src_mask() == [False, False, False, False])
        assert all(c.src_mask(src_type='gpot', dest_type='gpot') == [False, False])
        assert all(c.src_mask(src_type='gpot', dest_type='spike') == [False, False])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [False, False])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [False, False])

    def test_src_mask_no_conn_no_spike_A(self):
        c = core.Connectivity(2, 0, 3, 3)
        all(c.src_mask() == [False, False])
        assert all(c.src_mask(src_type='gpot', dest_type='gpot') == [False, False])
        assert all(c.src_mask(src_type='gpot', dest_type='spike') == [False, False])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [])
        
    def test_src_mask_conn(self):
        c = core.Connectivity(2, 2, 3, 3)
        c['A', 'gpot', 0, 'B', 'gpot', 1] = 1
        assert all(c.src_mask(src_type='gpot', dest_type='gpot') == [True, False])
        assert all(c.src_mask(src_type='gpot', dest_type='spike') == [False, False])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [False, False])        
        assert all(c.src_mask(src_type='spike', dest_type='spike') == [False, False])

    def test_src_mask_conn_slice(self):
        c = core.Connectivity(2, 2, 3, 3)
        c['A', 'all', :, 'B', 'gpot', 1] = 1
        assert all(c.src_mask(src_type='gpot', dest_type='gpot') == [True, True])
        assert all(c.src_mask(src_type='gpot', dest_type='spike') == [False, False])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [True, True])        
        assert all(c.src_mask(src_type='spike', dest_type='spike') == [False, False])
        
    def test_dest_mask_no_conn(self):
        c = core.Connectivity(2, 2, 3, 3)
        all(c.dest_mask() == [False, False, False, False, False, False])
        assert all(c.dest_mask(src_type='gpot', dest_type='gpot') == [False, False, False])        
        assert all(c.dest_mask(src_type='gpot', dest_type='spike') == [False, False, False])        
        assert all(c.dest_mask(src_type='spike', dest_type='gpot') == [False, False, False])
        assert all(c.dest_mask(src_type='spike', dest_type='gpot') == [False, False, False])
        
    def test_dest_mask_conn(self):
        c = core.Connectivity(2, 2, 3, 3)
        c['A', 'gpot', 0, 'B', 'gpot', 1] = 1
        assert all(c.dest_mask(src_type='gpot', dest_type='gpot') == [False, True, False])
        assert all(c.dest_mask(src_type='gpot', dest_type='spike') == [False, False, False])
        assert all(c.dest_mask(src_type='spike', dest_type='gpot') == [False, False, False])        
        assert all(c.dest_mask(src_type='spike', dest_type='spike') == [False, False, False])
        
if __name__ == '__main__':
    main()
