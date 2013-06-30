#!/usr/bin/env python

"""
Unit tests for neurokernel.core.
"""

from unittest import main, TestCase, TestSuite

import neurokernel.core as core

class test_Connectivity(TestCase):
    def test_Connectivity_src_mask_no_conn(self):
        c = core.Connectivity(2, 2, 3, 3)
        all(c.src_mask() == [False, False, False, False])
        assert all(c.src_mask(src_type='gpot', dest_type='gpot') == [False, False])
        assert all(c.src_mask(src_type='gpot', dest_type='spike') == [False, False])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [False, False])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [False, False])
        
    def test_Connectivity_src_mask_conn(self):
        c = core.Connectivity(2, 2, 3, 3)
        c['A', 'gpot', 0, 'B', 'gpot', 1] = 1
        assert all(c.src_mask(src_type='gpot', dest_type='gpot') == [True, False])
        assert all(c.src_mask(src_type='gpot', dest_type='spike') == [False, False])
        assert all(c.src_mask(src_type='spike', dest_type='gpot') == [False, False])        
        assert all(c.src_mask(src_type='spike', dest_type='spike') == [False, False])

    def test_Connectivity_dest_mask_no_conn(self):
        c = core.Connectivity(2, 2, 3, 3)
        all(c.dest_mask() == [False, False, False, False, False, False])
        assert all(c.dest_mask(src_type='gpot', dest_type='gpot') == [False, False, False])        
        assert all(c.dest_mask(src_type='gpot', dest_type='spike') == [False, False, False])        
        assert all(c.dest_mask(src_type='spike', dest_type='gpot') == [False, False, False])
        assert all(c.dest_mask(src_type='spike', dest_type='gpot') == [False, False, False])
        
    def test_Connectivity_dest_mask_conn(self):
        c = core.Connectivity(2, 2, 3, 3)
        c['A', 'gpot', 0, 'B', 'gpot', 1] = 1
        assert all(c.dest_mask(src_type='gpot', dest_type='gpot') == [False, True, False])
        assert all(c.dest_mask(src_type='gpot', dest_type='spike') == [False, False, False])
        assert all(c.dest_mask(src_type='spike', dest_type='gpot') == [False, False, False])        
        assert all(c.dest_mask(src_type='spike', dest_type='spike') == [False, False, False])
        
if __name__ == '__main__':
    main()
