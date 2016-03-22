#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
import pandas as pd
import networkx as nx
from pandas.util.testing import assert_frame_equal, assert_index_equal, \
    assert_series_equal

from neurokernel.pattern import Interface, Pattern, are_compatible

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

    def test_assign_unequal_levels(self):
        i = Interface('/a/b/c,/x/y')

        # This should succeed without exception:
        i['/x/y', 'interface'] = 0
        
    def test_create_dup_identifiers(self):
        self.assertRaises(Exception, Interface, '/foo[0],/foo[0]')

    def test_equals(self):
        i = Interface('/foo[0:2],/bar[0:2]')
        i['/foo[0]'] = [0, 'in', 'gpot']
        i['/bar[0]'] = [0, 'out', 'gpot']
        i['/foo[1]'] = [1, 'in', 'spike']
        i['/bar[1]'] = [1, 'out', 'spike']
        j = Interface('/foo[0:2],/bar[0:2]')
        j['/foo[0]'] = [0, 'in', 'gpot']
        j['/bar[0]'] = [0, 'out', 'gpot']
        j['/foo[1]'] = [1, 'in', 'spike']
        j['/bar[1]'] = [1, 'out', 'spike']
        assert i.equals(j)
        assert j.equals(i)
        j['/foo[0]'] = [0, 'in', 'spike']
        assert not i.equals(j)
        assert not j.equals(i)

    def test_get_common_ports(self):
        # Without type, single level:
        i = Interface('/foo,/bar,/baz')
        i['/*', 'interface'] = 0
        j = Interface('/bar,/baz,/qux')
        j['/*', 'interface'] = 0
        assert i.get_common_ports(0, j, 0, 'spike') == []
        self.assertItemsEqual(i.get_common_ports(0, j, 0),
                              [('bar',), ('baz',)])

        # Without type, multiple levels:
        i = Interface('/foo[0:6]')
        i['/*', 'interface'] = 0
        j = Interface('/foo[3:9]')
        j['/*', 'interface'] = 0
        assert i.get_common_ports(0, j, 0, 'spike') == []
        self.assertItemsEqual(i.get_common_ports(0, j, 0),
                              [('foo', 3), ('foo', 4), ('foo', 5)])

        # With type, single level:
        i = Interface('/foo,/bar,/baz')
        i['/foo,/bar', 'interface', 'type'] = [0, 'spike']
        j = Interface('/bar,/baz,/qux')
        j['/bar,/baz', 'interface', 'type'] = [0, 'spike']
        self.assertItemsEqual(i.get_common_ports(0, j, 0, 'spike'),
                              [('bar',)])
        
        # With type, multiple levels:
        i = Interface('/foo[0:6]')
        i['/foo[3,4]', 'interface', 'type'] = [0, 'spike']
        j = Interface('/foo[3:9]')
        j['/foo[3,4]', 'interface', 'type'] = [0, 'spike']
        self.assertItemsEqual(i.get_common_ports(0, j, 0, 'spike'),
                              [('foo', 3), ('foo', 4)])


    def test_get_common_ports_unequal_num_levels(self):
        # Without type, some with only one level:
        i = Interface('/foo[0:6],/bar')
        i['/*', 'interface'] = 0
        j = Interface('/bar,/baz')
        j['/*', 'interface'] = 0
        assert i.get_common_ports(0, j, 0, 'spike') == []
        self.assertItemsEqual(i.get_common_ports(0, j, 0),
                              [('bar',)])

        # Without type, more than one level:
        i = Interface('/foo[0:6],/bar[0:2]/baz')
        i['/*', 'interface'] = 0
        j = Interface('/foo[3:9]')
        j['/*', 'interface'] = 0
        assert i.get_common_ports(0, j, 0, 'spike') == []
        self.assertItemsEqual(i.get_common_ports(0, j, 0),
                              [('foo', 3), ('foo', 4), ('foo', 5)])

        # With type, some with only one level:
        i = Interface('/foo[0:6],/bar,/baz')
        i['/foo[3,4],/bar', 'interface', 'type'] = [0, 'spike']
        j = Interface('/bar,/baz,/qux')
        j['/bar,/baz', 'interface', 'type'] = [0, 'spike']
        self.assertItemsEqual(i.get_common_ports(0, j, 0, 'spike'),
                              [('bar',)])

        # With type, more than one level:
        i = Interface('/foo[0:6],/bar[0:2]/baz')
        i['/foo[3,4]', 'interface', 'type'] = [0, 'spike']
        j = Interface('/foo[3:9]')
        j['/foo[3,4]', 'interface', 'type'] = [0, 'spike']
        self.assertItemsEqual(i.get_common_ports(0, j, 0, 'spike'),
                              [('foo', 3), ('foo', 4)])

    def test_to_selectors(self):
        # Selector with multiple levels:
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

        # Selector with single level:
        i = Interface('/[foo,bar,baz]')
        i['/foo', 'interface'] = 0
        i['/bar', 'interface'] = 0
        i['/baz', 'interface'] = 1
        self.assertSequenceEqual(i.to_selectors(0),
                                 ['/foo', 
                                  '/bar'])
        self.assertSequenceEqual(i.to_selectors(), 
                                 ['/foo', 
                                  '/bar',
                                  '/baz'])

    def test_to_tuples_multi_levels(self):
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

    def test_to_tuples_single_level(self):
        i = Interface('[0:4]')
        i['[0:2]', 'interface'] = 0
        i['[2:4]', 'interface'] = 1
        self.assertSequenceEqual(i.to_tuples(0),
                                 [(0,), (1,)])
        self.assertSequenceEqual(i.to_tuples(),
                                 [(0,), (1,), (2,), (3,)])

    def test_data_select(self):
        # Selector with multiple levels:
        i = Interface('/foo[0:3]')
        i['/foo[0]', 'interface', 'io'] = [0, 'in']
        i['/foo[1:3]', 'interface', 'io'] = [0, 'out']
        j = i.data_select(lambda x: x['io'] != 'in')
        assert_index_equal(j.data.index,
                           pd.MultiIndex.from_tuples([('foo', 1),
                                                      ('foo', 2)],
                                                     names=[0, 1]))

        # Selector with single level:
        i = Interface('/[foo,bar,baz]')
        i['/[foo,bar]', 'interface', 'io'] = [0, 'in']
        i['/baz', 'interface', 'io'] = [0, 'out']
        j = i.data_select(lambda x: x['io'] != 'in')
        assert_index_equal(j.data.index,
                           pd.Index(['baz'], name=0))

        # Selectors with different numbers of levels:
        i = Interface('/a/b/c,/x/y')
        i['/a/b/c', 'interface', 'io'] = [0, 'in']
        j = i.data_select(lambda x: x['io'] != 'in')
        assert_index_equal(j.data.index,
                           pd.MultiIndex.from_tuples([('x', 'y', '')],
                                                     names=[0, 1, 2]))

    def test_from_df_index(self):
        idx = pd.Index(['foo', 'bar', 'baz'])
        data = [(0, 'in', 'spike'),
                (1, 'in', 'gpot'),
                (1, 'out', 'gpot')]
        columns = ['interface', 'io', 'type']
        df = pd.DataFrame(data, index=idx, columns=columns)
        i = Interface.from_df(df)
        assert_index_equal(i.data.index, idx)
        assert_frame_equal(i.data, df)

    def test_from_df_index_empty(self):
        idx = pd.Index([])
        data = None
        columns = ['interface', 'io', 'type']
        df = pd.DataFrame(data, index=idx, columns=columns)
        i = Interface.from_df(df)
        assert_index_equal(i.data.index, idx)
        assert_frame_equal(i.data, df)

    def test_from_df_multiindex(self):
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

    def test_from_df_multiindex_empty(self):
        idx = pd.MultiIndex(levels=[['a', 'b'], [0, 1]],
                            labels=[[],[]])
        data = None
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
                                                      ('foo', 2)],
                                                     names=[0, 1]))

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
        # Selector with multiple levels:
        i = Interface('/foo[0:3]')
        i['/foo[0]', 'interface', 'io'] = [0, 'in']
        i['/foo[1:3]', 'interface', 'io'] = [0, 'out']
        assert i.is_in_interfaces('/foo[0:3]') == True
        assert i.is_in_interfaces('/foo[0:4]') == False
        assert i.is_in_interfaces('/foo') == False

        # Selector with single level:
        i = Interface('/[foo,bar]')
        i['/foo', 'interface', 'io'] = [0, 'in']
        i['/bar', 'interface', 'io'] = [1, 'out']
        assert i.is_in_interfaces('/foo') == True
        assert i.is_in_interfaces('/qux') == False

        # Selectors comprising identifiers with different numbers of levels
        i = Interface('/foo,/bar[0:3]')
        i['/foo', 'interface', 'io'] = [0, 'in']
        i['/bar[0:3]', 'interface', 'io'] = [1, 'out']
        assert i.is_in_interfaces('/foo') == True
        assert i.is_in_interfaces('/bar[0]') == True
        assert i.is_in_interfaces('/bar') == False
        
    def test_in_ports(self):
        # Selector with multiple levels:
        i = Interface('/foo[0:2]')
        i['/foo[0]'] = [0, 'in', 'spike']
        i['/foo[1]'] = [1, 'out', 'spike']
        df = pd.DataFrame([(0, 'in', 'spike')],
                          pd.MultiIndex.from_tuples([('foo', 0)],
                                                    names=[0, 1]),
                          ['interface', 'io', 'type'],
                          dtype=object)

        # Test returning result as Interface:
        assert_frame_equal(i.in_ports(0).data, df)

        # Test returning result as list of tuples:
        self.assertItemsEqual(i.in_ports(0, True), df.index.tolist())

        # Selector with single level:
        i = Interface('/[foo,bar]')
        i['/foo'] = [0, 'in', 'spike']
        i['/bar'] = [1, 'out', 'spike']
        df = pd.DataFrame([(0, 'in', 'spike')],
                          pd.MultiIndex.from_tuples([('foo',)],
                                                    names=[0]),
                          ['interface', 'io', 'type'],
                          dtype=object)

        # Test returning result as Interface:
        assert_frame_equal(i.in_ports(0).data, df)

        # Test returning result as list of tuples:
        self.assertItemsEqual(i.in_ports(0, True), df.index.tolist())

    def test_interface_ports(self):
        # Selector with multiple levels:
        i = Interface('/foo[0:4]')
        i['/foo[0:2]', 'interface'] = 0
        i['/foo[2:4]', 'interface'] = 1
        j = Interface('/foo[2:4]')
        j['/foo[2:4]', 'interface'] = 1

        # Test returning result as Interface:
        assert_frame_equal(i.interface_ports(1).data, j.data)

        # Test returning result as list of tuples:
        self.assertItemsEqual(i.interface_ports(1, True), j.data.index.tolist())

        # Selector with single level:
        i = Interface('/[foo,bar,baz]')
        i['/[foo,bar]', 'interface'] = 0
        i['/baz', 'interface'] = 1
        j = Interface('/baz')
        j['/baz', 'interface'] = 1

        # Test returning result as Interface:
        assert_frame_equal(i.interface_ports(1).data, j.data)

        # Test returning result as list of tuples:
        self.assertItemsEqual(i.interface_ports(1, True), j.data.index.tolist())

    def test_out_ports(self):
        # Selector with multiple levels:
        i = Interface('/foo[0:2]')
        i['/foo[0]'] = [0, 'in', 'spike']
        i['/foo[1]'] = [1, 'out', 'spike']
        df = pd.DataFrame([(1, 'out', 'spike')],
                          pd.MultiIndex.from_tuples([('foo', 1)],
                                                    names=[0, 1]),
                          ['interface', 'io', 'type'],
                          dtype=object)

        # Test returning result as Interface:
        assert_frame_equal(i.out_ports(1).data, df)

        # Test returning result as list of tuples:
        self.assertItemsEqual(i.out_ports(1, True), df.index.tolist())

        # Selector with single level:
        i = Interface('/[foo,bar]')
        i['/foo'] = [0, 'in', 'spike']
        i['/bar'] = [1, 'out', 'spike']
        df = pd.DataFrame([(1, 'out', 'spike')],
                          pd.MultiIndex.from_tuples([('bar',)],
                                                    names=[0]),
                          ['interface', 'io', 'type'],
                          dtype=object)

        # Test returning result as Interface:
        assert_frame_equal(i.out_ports(1).data, df)

        # Test returning result as list of tuples:
        self.assertItemsEqual(i.out_ports(1, True), df.index.tolist())

    def test_gpot_ports(self):
        i = Interface('/foo[0:6]')
        i['/foo[0]'] = [0, 'in', 'spike']
        i['/foo[1:3]'] = [0, 'out', 'spike']
        i['/foo[3]'] = [0, 'in', 'gpot']
        i['/foo[4:6]'] = [0, 'out', 'gpot']
        j = Interface('/foo[3:6]')
        j['/foo[3]'] = [0, 'in', 'gpot']
        j['/foo[4:6]'] = [0, 'out', 'gpot']

        # Test returning result as Interface:
        assert_frame_equal(i.gpot_ports(0).data, j.data)

        # Test returning result as list of tuples:
        self.assertItemsEqual(i.gpot_ports(0, True),
                              j.data.index.tolist())

    def test_spike_ports(self):
        i = Interface('/foo[0:6]')
        i['/foo[0]'] = [0, 'in', 'spike']
        i['/foo[1:3]'] = [0, 'out', 'spike']
        i['/foo[3]'] = [0, 'in', 'gpot']
        i['/foo[4:6]'] = [0, 'out', 'gpot']
        j = Interface('/foo[0:3]')
        j['/foo[0]'] = [0, 'in', 'spike']
        j['/foo[1:3]'] = [0, 'out', 'spike']

        # Return result as Interface:
        assert_frame_equal(i.spike_ports(0).data, j.data)

        # Test returning result as list of tuples:
        self.assertItemsEqual(i.spike_ports(0, True),
                              j.data.index.tolist())
        
    def test_port_select(self):
        i = self.interface.port_select(lambda x: x[1] >= 1)
        assert_index_equal(i.data.index,
                           pd.MultiIndex.from_tuples([('foo', 1),
                                                      ('foo', 2)],
                                                     names=[0, 1]))

    def test_index(self):
        assert_index_equal(self.interface.index,
                           pd.MultiIndex(levels=[['foo'], [0, 1, 2]],
                                         labels=[[0, 0, 0], [0, 1, 2]],
                                         names=[0, 1]))

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

    def test_is_compatible_sel_order(self):
        """
        Interfaces with individually compatible ports that are ordered in
        different ways should still be deemed compatible.
        """

        # Selectors with multiple levels:
        i = Interface('/foo[0:2],/bar[0:2]')
        i['/foo[0:2]', 'interface', 'io'] = [0, 'in']
        i['/bar[0:2]', 'interface', 'io'] = [0, 'out']
        j = Interface('/bar[0:2],/foo[0:2]')
        j['/bar[0:2]', 'interface', 'io'] = [1, 'in']
        j['/foo[0:2]', 'interface', 'io'] = [1, 'out']
        assert i.is_compatible(0, j, 1)

        # Selectors with single level:
        i = Interface('/foo,/bar,/baz,/qux')
        i['/[foo,bar]', 'interface', 'io'] = [0, 'in']
        i['/[baz,qux]', 'interface', 'io'] = [0, 'out']
        j = Interface('/bar,/foo,/qux,/baz')
        j['/[baz,qux]', 'interface', 'io'] = [1, 'in']
        j['/[foo,bar]', 'interface', 'io'] = [1, 'out']
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_both_dirs(self):
        """
        It should be possible to define compatible interfaces containing both
        input and output ports.
        """

        i = Interface('/foo[0:4]')
        i['/foo[0:2]', 'interface', 'io'] = [0, 'out']
        i['/foo[2:4]', 'interface', 'io'] = [0, 'in']
        j = Interface('/foo[0:4]')
        j['/foo[0:2]', 'interface', 'io'] = [1, 'in']
        j['/foo[2:4]', 'interface', 'io'] = [1, 'out']
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_both_dirs_types(self):
        """
        It should be possible to define compatible interfaces containing both
        input and output ports with specified types.
        """

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
        """
        Interfaces can be compatible even if some of their ports do not have a
        set input or output status.
        """

        i = Interface('/foo[0:3]')
        i['/foo[0:2]', 'interface', 'io'] = [0, 'out']
        i['/foo[2]', 'interface'] = 0
        j = Interface('/foo[0:3]')
        j['/foo[0:2]', 'interface', 'io'] = [1, 'in']
        j['/foo[2]', 'interface'] = 1
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_with_nulls_types(self):
        """
        Interfaces can be compatible even if some of their ports do not have a
        set type.
        """

        i = Interface('/foo[0:3]')
        i['/foo[0:2]'] = [0, 'out', 'gpot']
        i['/foo[2]', 'interface'] = 0
        j = Interface('/foo[0:3]')
        j['/foo[0:2]'] = [1, 'in', 'gpot']
        j['/foo[2]', 'interface'] = 1
        assert i.is_compatible(0, j, 1)

    def test_is_compatible_subsets(self):
        """
        Interfaces that both share a subset of compatible ports can be deemed
        compatible by setting the `allow_subsets` option of the compatibility
        test.
        """

        i = Interface('/foo[0:6]')
        i['/foo[0:3]'] = [0, 'out', 'gpot']
        i['/foo[3:6]'] = [0, 'out', 'spike']
        j = Interface('/foo[0:6]')
        j['/foo[0:2]'] = [1, 'in', 'gpot']
        j['/foo[3:5]'] = [1, 'in', 'spike']
        k = Interface('/foo[0:6]')
        assert i.is_compatible(0, j, 1, True)
        assert i.is_compatible(0, k, 1, True) == False

    def test_is_compatible_subsets_with_null_types(self):
        """
        Interfaces that both share a subset of compatible ports can be deemed
        compatible by setting the `allow_subsets` option of the compatibility
        test even when the types are null.
        """

        i = Interface('/foo[0:6]')
        i['/foo[0:3]'] = [0, 'out']
        i['/foo[3:6]'] = [0, 'out']
        j = Interface('/foo[0:6]')
        j['/foo[0:2]'] = [1, 'in']
        j['/foo[3:5]'] = [1, 'in']
        k = Interface('/foo[0:6]')
        assert i.is_compatible(0, j, 1, True)
        assert i.is_compatible(0, k, 1, True) == False

    def test_are_compatible(self):
        assert are_compatible('/foo[2:4]', '/foo[0:2]', '/foo[2:4]', '/foo[0:2]',
                              '/foo[0:2]', '/foo[2:4]', '/foo[2:4]', '/foo[0:2]')

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
        self.df_p.sort_index(inplace=True)

        self.df_i = \
            pd.DataFrame(data={'interface': np.array([0, 0, 0, 0, 0, 
                                                      1, 1, 1, 1, 1], dtype=object),
                               'io': ['in', 'in', 'out', 'out', 'out', 
                                      'out', 'out', 'out', 'in', 'in'],
                               'type': ['spike', 'spike', 'gpot', 'gpot', 'gpot',
                                        'spike', 'spike', np.nan, 'gpot', 'gpot'],
                               0: ['foo', 'foo', 'foo', 'foo', 'foo',
                                   'bar', 'bar', 'bar', 'bar', 'bar'],
                               1: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]})
        self.df_i.set_index(0, append=False, inplace=True)
        self.df_i.set_index(1, append=True, inplace=True)

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

    def test_create_fan_in(self):
        pat = Pattern('/x[0:3]', '/y[0:3]')
        pat['/x[0]', '/y[0:2]'] = 1 # fan-out is allowed
        try:
            pat['/x[1:3]', '/y[2]'] = 1 # fan-in is not allowed
        except:
            pass
        else:
            raise Exception

    def test_create_unequal_levels(self):
        p = Pattern('/x[0:3]/y', '/z[0:3]')
        p['/x[0]/y', '/z[0]'] = 1

        q = Pattern('/x[0:3]', '/z[0:3]/a')
        q['/x[0]', '/z[0]/a'] = 1

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

        q = Pattern('/[aaa,bbb]', '/[www,xxx,yyy]')
        q['/aaa','/www'] = 1
        q['/aaa','/xxx'] = 1
        q['/bbb','/yyy'] = 1
        self.assertItemsEqual(q.src_idx(0, 1),
                              [('aaa',),
                               ('bbb',)])

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

        q = Pattern('/[aaa,bbb]', '/[www,xxx,yyy]')
        q['/aaa','/www'] = 1
        q['/aaa','/xxx'] = 1
        q['/bbb','/yyy'] = 1
        self.assertItemsEqual(q.src_idx(0, 1, dest_ports='/[www,xxx]'),
                              [('aaa',)])

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

        q = Pattern('/[aaa,bbb,ccc]', '/[www,xxx,yyy,zzz]')
        q['/aaa','/www'] = 1
        q['/aaa','/xxx'] = 1
        q['/yyy','/bbb'] = 1
        q['/zzz','/ccc'] = 1
        q.interface['/aaa'] = [0, 'in', 'spike']
        q.interface['/[www,xxx]'] = [1, 'out', 'spike']
        self.assertItemsEqual(q.src_idx(0, 1, src_type='spike'), 
                              [('aaa',)])
        self.assertItemsEqual(q.src_idx(0, 1, src_type='gpot'), [])

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

        q = Pattern('/[aaa,bbb,ccc]', '/[www,xxx,yyy,zzz]')
        q['/aaa','/www'] = 1
        q['/aaa','/xxx'] = 1
        q['/yyy','/bbb'] = 1
        q['/zzz','/ccc'] = 1
        q.interface['/aaa'] = [0, 'in', 'spike']
        q.interface['/[www,xxx]'] = [1, 'out', 'spike']
        self.assertItemsEqual(q.src_idx(0, 1, dest_type='spike'), 
                              [('aaa',)])
        self.assertItemsEqual(q.src_idx(0, 1, dest_type='gpot'), [])

    def test_src_idx_duplicates(self):
        p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
        p['/aaa[0]', '/yyy[0]'] = 1
        p['/aaa[0]', '/yyy[1]'] = 1
        p['/aaa[0]', '/yyy[2]'] = 1
        p['/xxx[0]', '/bbb[0]'] = 1
        p['/xxx[0]', '/bbb[1]'] = 1
        p['/xxx[1]', '/bbb[2]'] = 1
        self.assertItemsEqual(p.src_idx(0, 1, duplicates=True),
                              [('aaa', 0), ('aaa', 0), ('aaa', 0)])
        self.assertItemsEqual(p.src_idx(1, 0, duplicates=True),
                              [('xxx', 0), ('xxx', 0), ('xxx', 1)])

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

        q = Pattern('/[aaa,bbb]', '/[www,xxx,yyy]')
        q['/aaa','/www'] = 1
        q['/aaa','/xxx'] = 1
        q['/bbb','/yyy'] = 1
        self.assertItemsEqual(q.dest_idx(0, 1),
                              [('www',),
                               ('xxx',),
                               ('yyy',)])

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

        q = Pattern('/[aaa,bbb]', '/[www,xxx,yyy]')
        q['/aaa','/www'] = 1
        q['/aaa','/xxx'] = 1
        q['/bbb','/yyy'] = 1
        self.assertItemsEqual(q.dest_idx(0, 1, src_ports='/aaa'),
                              [('www',),
                               ('xxx',)])

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
        self.assertItemsEqual(p.dest_idx(0, 1, src_type='gpot'), [])

        q = Pattern('/[aaa,bbb,ccc]', '/[www,xxx,yyy,zzz]')
        q['/aaa','/www'] = 1
        q['/aaa','/xxx'] = 1
        q['/yyy','/bbb'] = 1
        q['/zzz','/ccc'] = 1
        q.interface['/aaa'] = [0, 'in', 'spike']
        q.interface['/[www,xxx]'] = [1, 'out', 'spike']
        self.assertItemsEqual(q.dest_idx(0, 1, src_type='spike'), 
                              [('www',),
                               ('xxx',)])
        self.assertItemsEqual(q.dest_idx(0, 1, src_type='gpot'), [])

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

        q = Pattern('/[aaa,bbb,ccc]', '/[www,xxx,yyy,zzz]')
        q['/aaa','/www'] = 1
        q['/aaa','/xxx'] = 1
        q['/yyy','/bbb'] = 1
        q['/zzz','/ccc'] = 1
        q.interface['/aaa'] = [0, 'in', 'spike']
        q.interface['/[www,xxx]'] = [1, 'out', 'spike']
        self.assertItemsEqual(q.dest_idx(0, 1, dest_type='spike'), 
                              [('www',),
                               ('xxx',)])
        self.assertItemsEqual(q.dest_idx(0, 1, dest_type='gpot'), [])

    def test_is_in_interfaces(self):
        # Selectors with multiple levels:
        p = Pattern('/aaa/bbb', '/ccc/ddd')
        assert p.is_in_interfaces('/aaa/bbb') == True
        assert p.is_in_interfaces('/aaa') == False

        # Selectors with a single level:
        p = Pattern('/aaa', '/bbb')
        assert p.is_in_interfaces('/aaa') == True
        assert p.is_in_interfaces('/ccc') == False

        # Selectors comprising identifiers with different numbers of levels::
        p = Pattern('/aaa,/bbb[0]', '/ccc,/ddd[0]')
        assert p.is_in_interfaces('/aaa') == True
        assert p.is_in_interfaces('/ccc') == True
        assert p.is_in_interfaces('/ddd') == False
        assert p.is_in_interfaces('/ddd[0]') == True

    def test_is_connected_single_level(self):

        # No connections:
        p = Pattern('/[aaa,bbb]', '/[ccc,ddd]')
        assert p.is_connected(0, 1) == False
        assert p.is_connected(1, 0) == False
        
        # Connected in one direction:
        p = Pattern('/[aaa,bbb]', '/[ccc,ddd]')
        p['/aaa', '/ccc'] = 1
        assert p.is_connected(0, 1) == True
        assert p.is_connected(1, 0) == False

        # Connected in both directions:
        p = Pattern('/[aaa,bbb,ccc]', '/[ddd,eee,fff]')
        p['/aaa', '/ddd'] = 1
        p['/eee', '/bbb'] = 1
        assert p.is_connected(0, 1) == True
        assert p.is_connected(1, 0) == True

    def test_is_connected_multi_level(self):
        
        # No connections:
        p = Pattern('/aaa[0:3]', '/bbb[0:3]')
        assert p.is_connected(0, 1) == False
        assert p.is_connected(1, 0) == False

        # Connected in one direction:
        p = Pattern('/aaa[0:3]', '/bbb[0:3]')
        p['/aaa[0]', '/bbb[2]'] = 1
        assert p.is_connected(0, 1) == True
        assert p.is_connected(1, 0) == False

        # Connected in both directions:
        p = Pattern('/aaa[0:3]', '/bbb[0:3]')
        p['/aaa[0]', '/bbb[2]'] = 1
        p['/bbb[0]', '/aaa[1]'] = 1
        assert p.is_connected(0, 1) == True
        assert p.is_connected(1, 0) == True

    def test_connected_port_pairs(self):
        p = Pattern('/aaa[0:3]', '/bbb[0:3]')
        p['/aaa[0]', '/bbb[2]'] = 1
        p['/aaa[1]', '/bbb[0]'] = 1
        p['/aaa[2]', '/bbb[1]'] = 1
        self.assertSequenceEqual(p.connected_port_pairs(),
                                 [(('aaa', 0), ('bbb', 2)),
                                  (('aaa', 1), ('bbb', 0)),
                                  (('aaa', 2), ('bbb', 1))])
        self.assertSequenceEqual(p.connected_port_pairs(True),
                                 [('/aaa/0', '/bbb/2'),
                                  ('/aaa/1', '/bbb/0'),
                                  ('/aaa/2', '/bbb/1')])

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

    def test_from_concat(self):
        # Need to specify selectors for both interfaces in pattern:
        self.assertRaises(ValueError, Pattern.from_concat, '', '/[baz,qux]',
                          from_sel='', to_sel='/[baz,qux]', data=1)

        # Patterns with interfaces using selectors with 1 level:
        p = Pattern.from_concat('/[foo,bar]', '/[baz,qux]',
                                from_sel='/[foo,bar]', to_sel='/[baz,qux]',
                                data=1)
        df = pd.DataFrame(data=[1, 1],
                          index=pd.MultiIndex(levels=[['bar', 'foo'], ['baz', 'qux']],
                                              labels=[[1, 0], [0, 1]],
                                              names=['from_0', 'to_0'], dtype=object),
                          columns=['conn'], dtype=object)
        assert_frame_equal(p.data, df)

        # Patterns with interfaces using selectors with more than 1 level:
        p = Pattern.from_concat('/foo[0:2]', '/bar[0:2]',
                                from_sel='/foo[0:2]', to_sel='/bar[0:2]',
                                data=1)
        df = pd.DataFrame(data=[1, 1],
                          index=pd.MultiIndex(levels=[['foo'], [0, 1], ['bar'], [0, 1]],
                                              labels=[[0, 0], [0, 1], [0, 0], [0, 1]],
                                              names=['from_0', 'from_1', 'to_0', 'to_1'], 
                                              dtype=object),
                          columns=['conn'], dtype=object)
        assert_frame_equal(p.data, df)

        # Patterns where port types are specified:
        p = Pattern.from_concat('/foo[0:2]', '/bar[0:2]',
                                from_sel='/foo[0:2]', to_sel='/bar[0:2]',
                                gpot_sel='/foo[0],/bar[0]',
                                spike_sel='/foo[1:2],/bar/[1:2]',
                                data=1)
        df_int = pd.DataFrame({'interface': [0, 0, 1, 1],
                               'io': ['in', 'in', 'out', 'out'],
                               'type': ['gpot', 'spike', 'gpot', 'spike']},
                              index=pd.MultiIndex(levels=[['bar', 'foo'], [0, 1]],
                                                  labels=[[1, 1, 0, 0], [0, 1, 0, 1]],
                                                  names=[0, 1],
                                                  dtype=object),
                              dtype=object)
        df = pd.DataFrame(data=[1, 1],
                          index=pd.MultiIndex(levels=[['foo'], [0, 1], ['bar'], [0, 1]],
                                              labels=[[0, 0], [0, 1], [0, 0], [0, 1]],
                                              names=['from_0', 'from_1', 'to_0', 'to_1'], 
                                              dtype=object),
                          columns=['conn'],
                          dtype=object)
        assert_frame_equal(p.data, df)
        assert_frame_equal(p.interface.data, df_int)

    def test_from_df(self):
        p = Pattern('/[aaa,bbb]/0', '/[ccc,ddd]/0')
        p['/aaa/0', '/ccc/0'] = 1
        p['/aaa/0', '/ddd/0'] = 1

        df_int = pd.DataFrame(data=[(0, 'in', np.nan),
                                    (0, np.nan, np.nan),
                                    (1, 'out', np.nan),
                                    (1, 'out', np.nan)],
                index=pd.MultiIndex(levels=[['aaa', 'bbb', 'ccc', 'ddd'], [0]], 
                                    labels=[[0, 1, 2, 3], [0, 0, 0, 0]],
                                    names=[0, 1]),
                              columns=['interface', 'io', 'type'],
                              dtype=object)
        df_pat = pd.DataFrame(data=[(1,), (1,)],
                index=pd.MultiIndex(levels=[['aaa'], [0], ['ccc', 'ddd'], [0]],
                                    labels=[[0, 0], [0, 0], [0, 1], [0, 0]],
                                    names=['from_0', 'from_1', 'to_0', 'to_1']),
                              columns=['conn'],
                              dtype=object)
        q = Pattern.from_df(df_int, df_pat)
        assert_frame_equal(p.data, q.data)
        assert_frame_equal(p.interface.data, q.interface.data)

    def test_to_graph(self):
        p = Pattern('/foo', '/bar')
        p['/foo', '/bar'] = 1
        g = p.to_graph()
        self.assertItemsEqual(g.nodes(data=True),
                              [('/foo', {'interface': 0, 'io': 'in', 'type': ''}),
                               ('/bar', {'interface': 1, 'io': 'out', 'type': ''})])

        p = Pattern('/foo[0:4]', '/bar[0:4]')
        p['/foo[0]', '/bar[0]'] = 1
        p['/foo[0]', '/bar[1]'] = 1
        p['/foo[1]', '/bar[2]'] = 1
        p['/bar[3]', '/foo[2]'] = 1
        p['/bar[3]', '/foo[3]'] = 1
        g = p.to_graph()

        self.assertItemsEqual(g.nodes(data=True), 
                              [('/bar/0', {'interface': 1, 'io': 'out', 'type': ''}),
                               ('/bar/1', {'interface': 1, 'io': 'out', 'type': ''}),
                               ('/bar/2', {'interface': 1, 'io': 'out', 'type': ''}),
                               ('/bar/3', {'interface': 1, 'io': 'in', 'type': ''}),
                               ('/foo/0', {'interface': 0, 'io': 'in', 'type': ''}),
                               ('/foo/1', {'interface': 0, 'io': 'in', 'type': ''}),
                               ('/foo/2', {'interface': 0, 'io': 'out', 'type': ''}),
                               ('/foo/3', {'interface': 0, 'io': 'out', 'type': ''})])
        self.assertItemsEqual(g.edges(data=True),
                              [('/foo/0', '/bar/0', {}),
                               ('/foo/0', '/bar/1', {}),
                               ('/foo/1', '/bar/2', {}),
                               ('/bar/3', '/foo/2', {}),
                               ('/bar/3', '/foo/3', {})])

        p.interface['/foo[0]','type'] = 'gpot'
        p.interface['/foo[1]','type'] = 'gpot'
        p.interface['/bar[0]','type'] = 'gpot'
        p.interface['/bar[1]','type'] = 'gpot'
        p.interface['/bar[2]','type'] = 'gpot'
        p.interface['/bar[3]','type'] = 'spike'
        p.interface['/foo[2]','type'] = 'spike'
        p.interface['/foo[3]','type'] = 'spike'
        g = p.to_graph()

        self.assertItemsEqual(g.nodes(data=True), 
                              [('/bar/0', {'interface': 1, 'io': 'out', 'type': 'gpot'}),
                               ('/bar/1', {'interface': 1, 'io': 'out', 'type': 'gpot'}),
                               ('/bar/2', {'interface': 1, 'io': 'out', 'type': 'gpot'}),
                               ('/bar/3', {'interface': 1, 'io': 'in', 'type': 'spike'}),
                               ('/foo/0', {'interface': 0, 'io': 'in', 'type': 'gpot'}),
                               ('/foo/1', {'interface': 0, 'io': 'in', 'type': 'gpot'}),
                               ('/foo/2', {'interface': 0, 'io': 'out', 'type': 'spike'}),
                               ('/foo/3', {'interface': 0, 'io': 'out', 'type': 'spike'})])

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
        assert_frame_equal(pg.data.sort_index(), p.data.sort_index())
        assert_frame_equal(pg.interface.data.sort_index(),
                           p.interface.data.sort_index())

        p.interface['/foo[0]', 'type'] = 'gpot'
        p.interface['/bar[0]', 'type'] = 'gpot'
        p.interface['/bar[1]', 'type'] = 'gpot'
        p.interface['/foo[1]', 'type'] = 'gpot'
        p.interface['/bar[2]', 'type'] = 'gpot'
        p.interface['/bar[3]', 'type'] = 'spike'
        p.interface['/foo[2]', 'type'] = 'spike'
        p.interface['/bar[3]', 'type'] = 'spike'
        p.interface['/foo[3]', 'type'] = 'spike'

        g = nx.DiGraph()
        g.add_node('/bar[0]', interface=1, io='out', type='gpot')
        g.add_node('/bar[1]', interface=1, io='out', type='gpot')
        g.add_node('/bar[2]', interface=1, io='out', type='gpot')
        g.add_node('/bar[3]', interface=1, io='in', type='spike')
        g.add_node('/foo[0]', interface=0, io='in', type='gpot')
        g.add_node('/foo[1]', interface=0, io='in', type='gpot')
        g.add_node('/foo[2]', interface=0, io='out', type='spike')
        g.add_node('/foo[3]', interface=0, io='out', type='spike')
        g.add_edge('/foo[0]', '/bar[0]')
        g.add_edge('/foo[0]', '/bar[1]')
        g.add_edge('/foo[1]', '/bar[2]')
        g.add_edge('/bar[3]', '/foo[2]')
        g.add_edge('/bar[3]', '/foo[3]')

        pg = Pattern.from_graph(g)
        assert_frame_equal(pg.data.sort_index(), p.data.sort_index())
        assert_frame_equal(pg.interface.data.sort_index(),
                           p.interface.data.sort_index())

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

    def test_connected_ports(self):
        p = Pattern('/foo[0:3]', '/bar[0:3]')
        p['/foo[0]', '/bar[0]'] = 1
        p['/foo[1]', '/bar[1]'] = 1
        self.assertItemsEqual(p.connected_ports(tuples=True),
                              [('bar', 0),
                               ('bar', 1),
                               ('foo', 0),
                               ('foo', 1)])
        self.assertItemsEqual(p.connected_ports(0, True),
                              [('foo', 0),
                               ('foo', 1)])
        self.assertItemsEqual(p.connected_ports(1, True),
                              [('bar', 0),
                               ('bar', 1)])

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

    def test_get_expanded(self):
        p = Pattern('/aaa[0:3]', '/bbb[0:3]')
        p['/aaa[0]', '/bbb[0]'] = 1
        p['/aaa[1]', '/bbb[1]'] = 1
        p['/aaa[2]', '/bbb[2]'] = 1
        df = pd.DataFrame({'conn': [1]},
            index=pd.MultiIndex(levels=[['aaa'], [0, 1, 2], ['bbb'], [0, 1, 2]],
                                labels=[[0], [0], [0], [0]],
                                names=['from_0', 'from_1', 'to_0', 'to_1']),
                          dtype=object)
        assert_frame_equal(p[[('aaa', 0)], [('bbb', 0)]], df)

if __name__ == '__main__':
    main()
