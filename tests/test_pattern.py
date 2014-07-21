#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
import pandas as pd
import networkx as nx
from pandas.util.testing import assert_frame_equal, assert_index_equal

from neurokernel.pattern import Interface, Pattern

class test_interface(TestCase):
    def setUp(self):
        self.interface = Interface('/foo[0:3]')
        self.interface['/foo[0]', 'interface', 'io'] = [0, 'in']
        self.interface['/foo[1:3]', 'interface', 'io'] = [0, 'out']

    def test_clear(self):
        i = Interface('/foo[0:4]')
        i.clear()
        assert len(i) == 0

    def test_create_empty(self):
        i = Interface('')
        assert len(i) == 0

    def test_create_dup_identifiers(self):
        self.assertRaises(Exception, Interface, '/foo[0],/foo[0]')

    def test_to_selectors(self):
        i = Interface('/foo[0:4]')
        i['/foo[0:2]', 'interface'] = 0
        i['/foo[2:4]', 'interface'] = 1
        self.assertSequenceEqual(i.to_selectors(0),
                                 ['/foo[0]', 
                                  '/foo[1]'])
        self.assertSequenceEqual(i.to_selectors(), 
                                 ['/foo[0]', 
                                  '/foo[1]',
                                  '/foo[2]',
                                  '/foo[3]'])

    def test_to_tuples(self):
        i = Interface('/foo[0:4]')
        i['/foo[0:2]', 'interface'] = 0
        i['/foo[2:4]', 'interface'] = 1
        self.assertSequenceEqual(i.to_tuples(0),
                                 [('foo', 0), 
                                  ('foo', 1)])
        self.assertSequenceEqual(i.to_tuples(), 
                                 [('foo', 0), 
                                  ('foo', 1),
                                  ('foo', 2),
                                  ('foo', 3)])

    def test_data_select(self):
        i = self.interface.data_select(lambda x: x['io'] >= 'out')
        assert_index_equal(i.data.index,
                           pd.MultiIndex.from_tuples([('foo', 1),
                                                      ('foo', 2)]))

    def test_from_df(self):
        idx = pd.MultiIndex.from_tuples([('foo', 0),
                                         ('foo', 1),
                                         ('foo', 2)])
        data = [(0, 'in', 'spike'),
                (1, 'in', 'gpot'),
                (1, 'out', 'gpot')]
        columns = ['interface', 'io', 'type']
        df = pd.DataFrame(data, index=idx, columns=columns)
        i = Interface.from_df(df)
        assert_index_equal(i.data.index, idx)
        assert_frame_equal(i.data, df)

    def test_from_df_dup(self):
        idx = pd.MultiIndex.from_tuples([('foo', 0),
                                         ('foo', 0),
                                         ('foo', 2)])
        data  = [(0, 'in', 'spike'),
                 (1, 'in', 'gpot'),
                 (1, 'out', 'gpot')]
        columns = ['interface', 'io', 'type']
        df = pd.DataFrame(data, index=idx, columns=columns)
        self.assertRaises(Exception, Interface.from_df, df)

    def test_from_dict(self):
        i = Interface.from_dict({'/foo[0:3]': np.nan})
        assert_index_equal(i.data.index,
                           pd.MultiIndex.from_tuples([('foo', 0),
                                                      ('foo', 1),
                                                      ('foo', 2)]))

    def test_from_graph(self):
        i = Interface('/foo[0:3]')
        i['/foo[0]'] = [0, 'in', 'gpot']
        i['/foo[1]'] = [0, 'out', 'gpot']
        i['/foo[2]'] = [0, 'out', 'spike']
        g = nx.Graph()
        g.add_node('/foo[0]', interface=0, io='in', type='gpot')
        g.add_node('/foo[1]', interface=0, io='out', type='gpot')
        g.add_node('/foo[2]', interface=0, io='out', type='spike')
        ig = Interface.from_graph(g)
        assert_index_equal(i.data.index, ig.data.index)
        assert_frame_equal(i.data, ig.data)

    def test_is_in_interfaces(self):
        assert self.interface.is_in_interfaces('/foo[0:3]') == True
        assert self.interface.is_in_interfaces('/foo[0:4]') == False

    def test_in_ports(self):
        i = Interface('/foo[0]')
        i['/foo[0]', 'interface', 'io'] = [0, 'in']
        assert_frame_equal(self.interface.in_ports(0).data, i.data)
        assert_index_equal(self.interface.in_ports(0).index, i.index)

    def test_interface_ports(self):
        i = Interface('/foo[0:4]')
        i['/foo[0:2]', 'interface'] = 0
        i['/foo[2:4]', 'interface'] = 1
        j = Interface('/foo[2:4]')
        j['/foo[2:4]', 'interface'] = 1
        assert_frame_equal(i.interface_ports(1).data, j.data)
        assert_index_equal(i.interface_ports(1).index, j.index)

    def test_out_ports(self):
        i = Interface('/foo[1:3]')
        i['/foo[1:3]', 'interface', 'io'] = [0, 'out']
        assert_frame_equal(self.interface.out_ports(0).data, i.data)
        assert_index_equal(self.interface.out_ports(0).index, i.index)

    def test_gpot_ports(self):
        i = Interface('/foo[0:6]')
        i['/foo[0]'] = [0, 'in', 'spike']
        i['/foo[1:3]'] = [0, 'out', 'spike']
        i['/foo[3]'] = [0, 'in', 'gpot']
        i['/foo[4:6]'] = [0, 'out', 'gpot']
        j = Interface('/foo[3:6]')
        j['/foo[3]'] = [0, 'in', 'gpot']
        j['/foo[4:6]'] = [0, 'out', 'gpot']
        assert_frame_equal(i.gpot_ports(0).data, j.data)
        assert_index_equal(i.gpot_ports(0).index, j.index)

    def test_spike_ports(self):
        i = Interface('/foo[0:6]')
        i['/foo[0]'] = [0, 'in', 'spike']
        i['/foo[1:3]'] = [0, 'out', 'spike']
        i['/foo[3]'] = [0, 'in', 'gpot']
        i['/foo[4:6]'] = [0, 'out', 'gpot']
        j = Interface('/foo[0:3]')
        j['/foo[0]'] = [0, 'in', 'spike']
        j['/foo[1:3]'] = [0, 'out', 'spike']
        assert_frame_equal(i.spike_ports(0).data, j.data)
        assert_index_equal(i.spike_ports(0).index, j.index)

    def test_port_select(self):
        i = self.interface.port_select(lambda x: x[1] >= 1)
        assert_index_equal(i.data.index,
                           pd.MultiIndex.from_tuples([('foo', 1),
                                                      ('foo', 2)]))

    def test_index(self):
        assert_index_equal(self.interface.index,
                           pd.MultiIndex(levels=[['foo'], [0, 1, 2]],
                                         labels=[[0, 0, 0], [0, 1, 2]],
                                         names=['0', '1']))

    def test_interface_ids(self):
        i = Interface('/foo[0:4]')
        i['/foo[0:2]', 'interface', 'io'] = [0, 'out']
        i['/foo[2:4]', 'interface', 'io'] = [1, 'in']
        assert i.interface_ids == set([0, 1])

    def test_io_inv(self):
        i = Interface('/foo[0:4]')
        i['/foo[0:2]', 'interface', 'io'] = [0, 'out']
        i['/foo[2:4]', 'interface', 'io'] = [1, 'in']
        j = Interface('/foo[0:4]')
        j['/foo[0:2]', 'interface', 'io'] = [0, 'in']
        j['/foo[2:4]', 'interface', 'io'] = [1, 'out']
        assert_frame_equal(i.data, j.io_inv.data)

    def test_is_compatible_both_dirs(self):
        i = Interface('/foo[0:4]')
        i['/foo[0:2]', 'interface', 'io'] = [0, 'out']
        i['/foo[2:4]', 'interface', 'io'] = [0, 'in']
        j = Interface('/foo[0:4]')
        j['/foo[0:2]', 'interface', 'io'] = [1, 'in']
        j['/foo[2:4]', 'interface', 'io'] = [1, 'out']
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_both_dirs_types(self):
        i = Interface('/foo[0:4]')
        i['/foo[0:2]'] = [0, 'out', 'gpot']
        i['/foo[2:4]'] = [0, 'in', 'spike']
        j = Interface('/foo[0:4]')
        j['/foo[0:2]'] = [1, 'in', 'gpot']
        j['/foo[2:4]'] = [1, 'out', 'spike']
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_one_dir(self):
        i = Interface('/foo[0:2]')
        i['/foo[0:2]', 'interface', 'io'] = [0, 'out']
        j = Interface('/foo[0:2]')
        j['/foo[0:2]', 'interface', 'io'] = [1, 'in']
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_one_dir_types(self):
        i = Interface('/foo[0:2]')
        i['/foo[0:2]'] = [0, 'out', 'spike']
        j = Interface('/foo[0:2]')
        j['/foo[0:2]'] = [1, 'in', 'spike']
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_with_nulls(self):
        i = Interface('/foo[0:3]')
        i['/foo[0:2]', 'interface', 'io'] = [0, 'out']
        i['/foo[2]', 'interface'] = 0
        j = Interface('/foo[0:3]')
        j['/foo[0:2]', 'interface', 'io'] = [1, 'in']
        j['/foo[2]', 'interface'] = 1
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_with_nulls_types(self):
        i = Interface('/foo[0:3]')
        i['/foo[0:2]'] = [0, 'out', 'gpot']
        i['/foo[2]', 'interface'] = 0
        j = Interface('/foo[0:3]')
        j['/foo[0:2]'] = [1, 'in', 'gpot']
        j['/foo[2]', 'interface'] = 1
        assert i.is_compatible(0, j, 1)

    def test_which_int_unset(self):
        i = Interface('/foo[0:4]')
        assert i.which_int('/foo[0:2]') == set()

    def test_which_int_set(self):
        i = Interface('/foo[0:4]')
        i['/foo[0]', 'interface', 'io'] = [0, 'out']
        i['/foo[1]', 'interface', 'io'] = [0, 'in']
        i['/foo[2]', 'interface', 'io'] = [1, 'in']
        i['/foo[3]', 'interface', 'io'] = [1, 'out']
        assert i.which_int('/foo[0:2]') == {0}
        assert i.which_int('/foo[0:4]') == {0, 1}

class test_pattern(TestCase):
    def setUp(self):
        self.df_p = pd.DataFrame(data={'conn': np.ones(6, dtype='object'),
                        'from_0': ['bar', 'bar', 'bar', 'foo', 'foo', 'foo'],
                        'from_1': [3, 3, 4, 0, 1, 1],
                        'to_0': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
                        'to_1': [2, 3, 4, 0, 1, 2]})
        self.df_p.set_index('from_0', append=False, inplace=True)
        self.df_p.set_index('from_1', append=True, inplace=True)
        self.df_p.set_index('to_0', append=True, inplace=True)
        self.df_p.set_index('to_1', append=True, inplace=True)
        self.df_p.sort(inplace=True)

        self.df_i = \
            pd.DataFrame(data={'interface': np.array([0, 0, 0, 0, 0, 
                                                      1, 1, 1, 1, 1], dtype=object),
                               'io': ['in', 'in', 'out', 'out', 'out', 
                                      'out', 'out', 'out', 'in', 'in'],
                               'type': ['spike', 'spike', 'gpot', 'gpot', 'gpot',
                                        'spike', 'spike', np.nan, 'gpot', 'gpot'],
                               '0': ['foo', 'foo', 'foo', 'foo', 'foo',
                                   'bar', 'bar', 'bar', 'bar', 'bar'],
                               '1': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]})
        self.df_i.set_index('0', append=False, inplace=True)
        self.df_i.set_index('1', append=True, inplace=True)

    def test_create(self):
        p = Pattern('/foo[0:5]', '/bar[0:5]')
        p['/foo[0]', '/bar[0]'] = 1
        p['/foo[1]', '/bar[1]'] = 1
        p['/foo[1]', '/bar[2]'] = 1
        p['/bar[3]', '/foo[2]'] = 1
        p['/bar[3]', '/foo[3]'] = 1
        p['/bar[4]', '/foo[4]'] = 1
        assert_frame_equal(p.data, self.df_p)
        p.interface['/foo[0:2]', 'type'] = 'spike'
        p.interface['/bar[0:2]', 'type'] = 'spike'
        p.interface['/foo[2:5]', 'type'] = 'gpot'
        p.interface['/bar[3:5]', 'type'] = 'gpot'
        assert_frame_equal(p.interface.data, self.df_i)

    def test_create_dup_identifiers(self):
        self.assertRaises(Exception,  Pattern,
                          '/foo[0],/foo[0]', '/bar[0:2]')

    def test_create_port_in_out(self):
        self.assertRaises(Exception,  Pattern,
                          '/[foo,bar][0]', '/bar[0:2]')

    def test_src_idx(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[1]', '/yyy[1]'] = 1
        p['/aaa[2]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[1]', '/bbb[1]'] = 1
        p['/xxx[2]', '/bbb[2]'] = 1
        self.assertItemsEqual(p.src_idx(0, 1),
                              [('aaa', 0),
                               ('aaa', 1),
                               ('aaa', 2)])

    def test_src_idx_dest_ports(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[0]', '/yyy[1]'] = 1
        p['/aaa[0]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[1]', '/bbb[1]'] = 1
        p['/xxx[2]', '/bbb[2]'] = 1
        self.assertItemsEqual(p.src_idx(0, 1, dest_ports='/yyy[0]'),
                              [('aaa', 0)])

    def test_src_idx_src_type(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[0]', '/yyy[1]'] = 1
        p['/aaa[0]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[1]', '/bbb[1]'] = 1
        p['/xxx[2]', '/bbb[2]'] = 1
        p.interface['/aaa[0:3]'] = [0, 'in', 'spike']
        p.interface['/yyy[0:3]'] = [1, 'out', 'spike']
        self.assertItemsEqual(p.src_idx(0, 1, src_type='spike'), 
                              [('aaa', 0)])
        self.assertItemsEqual(p.src_idx(0, 1, src_type='gpot'), [])

    def test_src_idx_dest_type(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[0]', '/yyy[1]'] = 1
        p['/aaa[0]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[1]', '/bbb[1]'] = 1
        p['/xxx[2]', '/bbb[2]'] = 1
        p.interface['/aaa[0:3]', 'type'] = 'spike'
        p.interface['/yyy[0:3]', 'type'] = 'spike'
        self.assertItemsEqual(p.src_idx(0, 1, dest_type='spike'), 
                              [('aaa', 0)])
        self.assertItemsEqual(p.src_idx(0, 1, dest_type='gpot'), [])

    def test_dest_idx(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[1]', '/yyy[1]'] = 1
        p['/aaa[2]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[1]', '/bbb[1]'] = 1
        p['/xxx[2]', '/bbb[2]'] = 1
        self.assertItemsEqual(p.dest_idx(0, 1),
                              [('yyy', 0),
                               ('yyy', 1),
                               ('yyy', 2)])

    def test_dest_idx_src_ports(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[0]', '/yyy[1]'] = 1
        p['/aaa[0]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[1]', '/bbb[1]'] = 1
        p['/xxx[2]', '/bbb[2]'] = 1
        self.assertItemsEqual(p.dest_idx(0, 1, src_ports='/aaa[0]'),
                              [('yyy', 0),
                               ('yyy', 1),
                               ('yyy', 2)])

    def test_dest_idx_src_type(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[0]', '/yyy[1]'] = 1
        p['/aaa[0]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[1]', '/bbb[1]'] = 1
        p['/xxx[2]', '/bbb[2]'] = 1
        p.interface['/aaa[0:3]'] = [0, 'in', 'spike']
        p.interface['/yyy[0:3]'] = [1, 'out', 'spike']
        self.assertItemsEqual(p.dest_idx(0, 1, src_type='spike'), 
                              [('yyy', 0),
                               ('yyy', 1),
                               ('yyy', 2)])
        self.assertItemsEqual(p.src_idx(0, 1, src_type='gpot'), [])

    def test_dest_idx_dest_type(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[0]', '/yyy[1]'] = 1
        p['/aaa[0]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[1]', '/bbb[1]'] = 1
        p['/xxx[2]', '/bbb[2]'] = 1
        p.interface['/aaa[0:3]', 'type'] = 'spike'
        p.interface['/yyy[0:3]', 'type'] = 'spike'
        self.assertItemsEqual(p.dest_idx(0, 1, dest_type='spike'), 
                              [('yyy', 0),
                               ('yyy', 1),
                               ('yyy', 2)])
        self.assertItemsEqual(p.dest_idx(0, 1, dest_type='gpot'), [])

    def test_is_connected(self):
        p = Pattern('/aaa[0:3]', '/bbb[0:3]')
        p['/aaa[0]', '/bbb[2]'] = 1
        assert p.is_connected(0, 1) == True
        assert p.is_connected(1, 0) == False

    def test_get_conns(self):
        p = Pattern('/aaa[0:3]', '/bbb[0:3]')
        p['/aaa[0]', '/bbb[2]'] = 1
        p['/aaa[1]', '/bbb[0]'] = 1
        p['/aaa[2]', '/bbb[1]'] = 1
        self.assertSequenceEqual(p.get_conns(),
                                 [(('aaa', 0), ('bbb', 2)),
                                  (('aaa', 1), ('bbb', 0)),
                                  (('aaa', 2), ('bbb', 1))])
        self.assertSequenceEqual(p.get_conns(True),
                                 [('/aaa[0]', '/bbb[2]'),
                                  ('/aaa[1]', '/bbb[0]'),
                                  ('/aaa[2]', '/bbb[1]')])

    def test_split_multiindex(self):
        idx = pd.MultiIndex(levels=[['a'], ['b', 'c'], ['d', 'e'], [0, 1, 2]],
                            labels=[[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 2]])
        idx0, idx1 = Pattern.split_multiindex(idx, slice(0, 2), slice(2, 4))
        assert_index_equal(idx0,
                           pd.MultiIndex(levels=[['a'], ['b', 'c']],
                                         labels=[[0, 0, 0, 0], [0, 0, 1, 1]]))
        assert_index_equal(idx1,
                           pd.MultiIndex(levels=[['d', 'e'], [0, 1, 2]],
                                         labels=[[0, 1, 0, 1], [0, 1, 1, 2]]))

        idx0, idx1 = Pattern.split_multiindex(idx, slice(0, 1), slice(1, 4))
        assert_index_equal(idx0,
                           pd.Index(['a', 'a', 'a', 'a']))
        assert_index_equal(idx1,
                           pd.MultiIndex(levels=[['b', 'c'], ['d', 'e'], [0, 1, 2]],
                                         labels=[[0, 0, 1, 1], [0, 1, 0, 1],
                                                 [0, 1, 1, 2]]))

    def test_to_graph(self):
        p = Pattern('/foo[0:4]', '/bar[0:4]')
        p['/foo[0]', '/bar[0]'] = 1
        p['/foo[0]', '/bar[1]'] = 1
        p['/foo[1]', '/bar[2]'] = 1
        p['/bar[3]', '/foo[2]'] = 1
        p['/bar[3]', '/foo[3]'] = 1
        g = p.to_graph()

        self.assertItemsEqual(g.nodes(data=True), 
                              [('/bar[0]', {'interface': 1, 'io': 'out', 'type': ''}),
                               ('/bar[1]', {'interface': 1, 'io': 'out', 'type': ''}),
                               ('/bar[2]', {'interface': 1, 'io': 'out', 'type': ''}),
                               ('/bar[3]', {'interface': 1, 'io': 'in', 'type': ''}),
                               ('/foo[0]', {'interface': 0, 'io': 'in', 'type': ''}),
                               ('/foo[1]', {'interface': 0, 'io': 'in', 'type': ''}),
                               ('/foo[2]', {'interface': 0, 'io': 'out', 'type': ''}),
                               ('/foo[3]', {'interface': 0, 'io': 'out', 'type': ''})])
        self.assertItemsEqual(g.edges(data=True),
                              [('/foo[0]', '/bar[0]', {}),
                               ('/foo[0]', '/bar[1]', {}),
                               ('/foo[1]', '/bar[2]', {}),
                               ('/bar[3]', '/foo[2]', {}),
                               ('/bar[3]', '/foo[3]', {})])

    def test_from_graph(self):
        p = Pattern('/foo[0:4]', '/bar[0:4]')
        p['/foo[0]', '/bar[0]'] = 1
        p['/foo[0]', '/bar[1]'] = 1
        p['/foo[1]', '/bar[2]'] = 1
        p['/bar[3]', '/foo[2]'] = 1
        p['/bar[3]', '/foo[3]'] = 1

        g = nx.DiGraph()
        g.add_node('/bar[0]', interface=1, io='out')
        g.add_node('/bar[1]', interface=1, io='out')
        g.add_node('/bar[2]', interface=1, io='out')
        g.add_node('/bar[3]', interface=1, io='in')
        g.add_node('/foo[0]', interface=0, io='in')
        g.add_node('/foo[1]', interface=0, io='in')
        g.add_node('/foo[2]', interface=0, io='out')
        g.add_node('/foo[3]', interface=0, io='out')
        g.add_edge('/foo[0]', '/bar[0]')
        g.add_edge('/foo[0]', '/bar[1]')
        g.add_edge('/foo[1]', '/bar[2]')
        g.add_edge('/bar[3]', '/foo[2]')
        g.add_edge('/bar[3]', '/foo[3]')

        pg = Pattern.from_graph(g)
        assert_frame_equal(pg.data.sort(), p.data.sort())
        assert_frame_equal(pg.interface.data.sort(), p.interface.data.sort())

    def test_gpot_ports(self):
        p = Pattern('/foo[0:3]', '/bar[0:3]')
        p.interface['/foo[0]', 'io', 'type'] = ['in', 'spike']
        p.interface['/foo[1:3]', 'io', 'type'] = ['out', 'gpot']
        p.interface['/bar[0:2]', 'io', 'type'] = ['out', 'spike']
        p.interface['/bar[2]', 'io', 'type'] = ['in', 'gpot']
        self.assertItemsEqual(p.gpot_ports(0).to_tuples(),
                              [('foo', 1),
                               ('foo', 2)])
        self.assertItemsEqual(p.gpot_ports(1).to_tuples(),
                              [('bar', 2)])

    def test_in_ports(self):
        p = Pattern('/foo[0:3]', '/bar[0:3]')
        p.interface['/foo[0]', 'io', 'type'] = ['in', 'spike']
        p.interface['/foo[1:3]', 'io', 'type'] = ['out', 'gpot']
        p.interface['/bar[0:2]', 'io', 'type'] = ['out', 'spike']
        p.interface['/bar[2]', 'io', 'type'] = ['in', 'gpot']
        self.assertItemsEqual(p.in_ports(0).to_tuples(),
                              [('foo', 0)])
        self.assertItemsEqual(p.in_ports(1).to_tuples(),
                              [('bar', 2)])

    def test_interface_ports(self):
        p = Pattern('/foo[0:3]', '/bar[0:3]')
        p.interface['/foo[0]', 'io', 'type'] = ['in', 'spike']
        p.interface['/foo[1:3]', 'io', 'type'] = ['out', 'gpot']
        p.interface['/bar[0:2]', 'io', 'type'] = ['out', 'spike']
        p.interface['/bar[2]', 'io', 'type'] = ['in', 'gpot']
        self.assertItemsEqual(p.interface_ports(0).to_tuples(),
                              [('foo', 0),
                               ('foo', 1),
                               ('foo', 2)])
        self.assertItemsEqual(p.interface_ports(1).to_tuples(),
                              [('bar', 0),
                               ('bar', 1),
                               ('bar', 2)])

    def test_out_ports(self): ###
        p = Pattern('/foo[0:3]', '/bar[0:3]')
        p.interface['/foo[0]', 'io', 'type'] = ['in', 'spike']
        p.interface['/foo[1:3]', 'io', 'type'] = ['out', 'gpot']
        p.interface['/bar[0:2]', 'io', 'type'] = ['out', 'spike']
        p.interface['/bar[2]', 'io', 'type'] = ['in', 'gpot']
        self.assertItemsEqual(p.out_ports(0).to_tuples(),
                              [('foo', 1),
                               ('foo', 2)])
        self.assertItemsEqual(p.out_ports(1).to_tuples(),
                              [('bar', 0),
                               ('bar', 1)])

    def test_clear(self):
        p = Pattern('/aaa[0:3]', '/bbb[0:3]')
        p['/aaa[0]', '/bbb[0]'] = 1
        p['/aaa[1]', '/bbb[1]'] = 1
        p['/aaa[2]', '/bbb[2]'] = 1
        p.clear()
        assert len(p) == 0
        assert len(p.interface) == 0

if __name__ == '__main__':
    main()
