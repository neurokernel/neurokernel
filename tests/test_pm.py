#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal, assert_index_equal, \
    assert_series_equal

from neurokernel.pm import BasePortMapper, PortMapper

class test_base_port_mapper(TestCase):
    def test_create(self):
        portmap = np.arange(5)
        pm = BasePortMapper('/foo[0:5]', portmap)
        s = pd.Series(np.arange(5),
                      pd.MultiIndex(levels=[['foo'], [0, 1, 2, 3, 4]],
                                    labels=[[0, 0, 0, 0, 0],
                                            [0, 1, 2, 3, 4]],
                                    names=[0, 1]))
        assert_series_equal(pm.portmap, s)

    def test_from_pm(self):
        # Ensure that modifying pm0 doesn't modify any other mapper created from it:
        pm0 = BasePortMapper('/foo[0:5]', np.arange(5))
        pm1 = BasePortMapper('/foo[0:5]', np.arange(5))
        pm2 = BasePortMapper.from_pm(pm0)
        pm0.portmap[('foo', 0)] = 10
        assert_series_equal(pm2.portmap, pm1.portmap)

    def test_copy(self):
        # Ensure that modifying pm0 doesn't modify any other mapper created from it:
        pm0 = BasePortMapper('/foo[0:5]', np.arange(5))
        pm1 = BasePortMapper('/foo[0:5]', np.arange(5))
        pm2 = pm0.copy()
        pm0.portmap[('foo', 0)] = 10
        assert_series_equal(pm2.portmap, pm1.portmap)

    def test_len(self):
        pm = BasePortMapper('/foo[0:5],/bar[0:5]')
        assert len(pm) == 10

    def test_equals(self):
        # Check that mappers containing the same ports/indices are deemed equal:
        pm0 = BasePortMapper('/foo[0:5],/bar[0:5]')
        pm1 = BasePortMapper('/foo[0:5],/bar[0:5]')
        assert pm0.equals(pm1)
        assert pm1.equals(pm0)

        # Check that mappers containing the same ports/indices in
        # different orders are deemed equal:
        pm0 = BasePortMapper('/foo[0:5],/bar[0:5]', list(range(10)))
        pm1 = BasePortMapper('/bar[0:5],/foo[0:5]', list(range(5, 10))+list(range(5)))
        assert pm0.equals(pm1)
        assert pm1.equals(pm0)

        # Check that mappers containing different ports/indices are deemed non-equal:
        pm0 = BasePortMapper('/foo[0:5],/bar[1:5]/bar[0]')
        pm1 = BasePortMapper('/foo[0:5],/bar[0:5]')
        assert not pm0.equals(pm1)
        assert not pm1.equals(pm0)

    def test_from_index(self):
        # Without a specified port map:
        pm0 = BasePortMapper('/foo[0:5],/bar[0:5]')
        pm1 = BasePortMapper.from_index(pm0.index)
        assert_series_equal(pm0.portmap, pm1.portmap)

        # With a specified port map:
        pm0 = BasePortMapper('/foo[0:5],/bar[0:5]', list(range(5))*2)
        pm1 = BasePortMapper.from_index(pm0.index, list(range(5))*2)
        assert_series_equal(pm0.portmap, pm1.portmap)

        # Ensure that modifying the map sequence used to create the
        # port mapper doesn't have the side effect of altering the created
        # mapper:
        index = pd.MultiIndex(levels=[[u'foo'], [0, 1, 2, 3, 4]],
                              labels=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]],
                              names=[0, 1])
        portmap = np.arange(5)
        pm1 = BasePortMapper.from_index(index, portmap)
        portmap[0] = 10
        assert_array_equal(pm1.portmap.values, np.arange(5))

    def test_inds_to_ports(self):
        # Without a specified port map:
        pm = BasePortMapper('/foo[0:5],/bar[0:5]')
        self.assertSequenceEqual(pm.inds_to_ports([4, 5]),
                                 [('foo', 4), ('bar', 0)])

        # With a specified port map:
        pm = BasePortMapper('/foo[0:5],/bar[0:5]', list(range(10, 20)))
        self.assertSequenceEqual(pm.inds_to_ports([14, 15]),
                                 [('foo', 4), ('bar', 0)])

    def test_ports_to_inds(self):
        # Without a specified port map:
        pm = BasePortMapper('/foo[0:5],/bar[0:5]')
        np.allclose(pm.ports_to_inds('/foo[4],/bar[0]'), [4, 5])

        # Nonexistent ports should return an empty index array:
        i = pm.ports_to_inds('/baz')
        assert len(i) == 0 and i.dtype == np.int_

        # With a specified port map:
        pm = BasePortMapper('/foo[0:5],/bar[0:5]', list(range(10, 20)))
        np.allclose(pm.ports_to_inds('/foo[4],/bar[0]'), [14, 15])

        i = pm.ports_to_inds('/baz')
        assert len(i) == 0 and i.dtype == np.int_

    def test_get_map(self):
        # Try to get selector that is in the mapper:
        pm = BasePortMapper('/foo[0:5],/bar[0:5]')
        self.assertSequenceEqual(pm.get_map('/bar[0:5]').tolist(), list(range(5, 10)))

        # Try to get selector that is not in the mapper:
        self.assertSequenceEqual(pm.get_map('/foo[5:10]').tolist(), [])

    def test_set_map(self):
        pm = BasePortMapper('/foo[0:5],/bar[0:5]')
        pm.set_map('/bar[0:5]', list(range(5)))
        self.assertSequenceEqual(pm.portmap.iloc[5:10].tolist(), list(range(5)))

class test_port_mapper(TestCase):
    def setUp(self):
        self.data = np.random.rand(20)

    def test_create(self):

        # Empty selector, empty data (force index dtype of ground truth to
        # object because neurokernel.plsel.SelectorMethods.make_index() creates
        # indexes with dtype=object):
        pm = PortMapper('')
        assert_series_equal(pm.portmap,
            pd.Series([], dtype=np.int_, index=pd.MultiIndex(levels=[[]], labels=[[]], names=[0])))
        assert_array_equal(pm.data, np.array([]))


        # Non-empty selector, empty data:
        pm = PortMapper('/foo[0:3]')
        assert_series_equal(pm.portmap,
                            pd.Series(np.arange(3),
                                      pd.MultiIndex(levels=[['foo'], [0, 1, 2]],
                                                    labels=[[0, 0, 0], [0, 1, 2]],
                                                    names=[0, 1])))
        assert_array_equal(pm.data, np.array([]))

        # Empty selector, non-empty data:
        # self.assertRaises(Exception, PortMapper, '', [1, 2, 3])

        # Non-empty selector, non-empty data:
        data = np.random.rand(5)
        portmap = np.arange(5)
        pm = PortMapper('/foo[0:5]', data, portmap)
        assert_array_equal(pm.data, data)
        s = pd.Series(np.arange(5),
                      pd.MultiIndex(levels=[['foo'], [0, 1, 2, 3, 4]],
                                    labels=[[0, 0, 0, 0, 0],
                                            [0, 1, 2, 3, 4]],
                                    names=[0, 1]))
        assert_series_equal(pm.portmap, s)

    def test_from_pm(self):
        # Ensure that modifying pm0 doesn't modify any other mapper created from it:
        data = np.random.rand(5)
        portmap = np.arange(5)
        pm0 = PortMapper('/foo[0:5]', data, portmap)
        pm1 = PortMapper('/foo[0:5]', data, portmap)
        pm2 = PortMapper.from_pm(pm0)
        data[0] = 1.0
        pm0.data[1] = 1.0
        pm0.portmap[('foo', 0)] = 10
        assert_array_equal(pm2.data, pm1.data)
        assert_series_equal(pm2.portmap, pm1.portmap)

    def test_copy(self):
        # Ensure that modifying pm0 doesn't modify any other mapper created from it:
        data = np.random.rand(5)
        portmap = np.arange(5)
        pm0 = PortMapper('/foo[0:5]', data, portmap)
        pm1 = PortMapper('/foo[0:5]', data, portmap)
        pm2 = pm0.copy()
        data[0] = 1.0
        pm0.data[1] = 1.0
        pm0.portmap[('foo', 0)] = 10
        assert_array_equal(pm2.data, pm1.data)
        assert_series_equal(pm2.portmap, pm1.portmap)

        data = np.random.rand(5)
        pm0 = PortMapper('/foo[0:5]', data, portmap, False)
        pm1 = pm0.copy()
        data[0] = 1.0
        assert pm0.data[0] == 1.0

    def test_dtype(self):
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        assert pm.dtype == np.float64

    def test_equals(self):
        pm0 = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        pm1 = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        assert pm0.equals(pm1)
        assert pm1.equals(pm0)
        pm0 = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        pm1 = PortMapper('/foo/bar[0:10],/foo/baz[1:10],/foo/baz[0]', self.data)
        assert not pm0.equals(pm1)
        assert not pm1.equals(pm0)
        pm0 = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', np.arange(20))
        pm1 = PortMapper('/foo/bar[0:10],/foo/baz[0:10]',
                         np.concatenate((np.arange(10), np.arange(10))))
        assert not pm0.equals(pm1)
        assert not pm1.equals(pm0)


    def test_get(self):
        # Mapper with data:
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        np.allclose(self.data[0:10], pm['/foo/bar[0:10]'])
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]')

        # Mapper without data:
        self.assertRaises(Exception, pm.__getitem__, '/foo/bar[0]')

    def test_get_discontinuous(self):
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        np.allclose(self.data[[0, 2, 4, 6]],
                    pm['/foo/bar[0,2,4,6]'])

    def test_get_sub(self):
        pm = PortMapper('/foo/bar[0:5],/foo/baz[0:5]', self.data,
                        np.arange(5, 15))
        np.allclose(self.data[5:10], pm['/foo/bar[0:5]'])

    def test_get_ports(self):
        pm = PortMapper('/foo/bar[0:10]', np.arange(10))
        self.assertSequenceEqual(pm.get_ports(lambda x: x < 5),
                                 [('foo', 'bar', 0),
                                  ('foo', 'bar', 1),
                                  ('foo', 'bar', 2),
                                  ('foo', 'bar', 3),
                                  ('foo', 'bar', 4)])
        i = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.bool)
        self.assertSequenceEqual(pm.get_ports(i),
                                 [('foo', 'bar', 0),
                                  ('foo', 'bar', 1),
                                  ('foo', 'bar', 2),
                                  ('foo', 'bar', 3),
                                  ('foo', 'bar', 4)])

    def test_get_ports_as_inds(self):
        pm = PortMapper('/foo[0:5]', np.array([0, 1, 0, 1, 0]))
        np.allclose(pm.get_ports_as_inds(lambda x: np.asarray(x, dtype=np.bool)),
                    [1, 3])

    def test_get_ports_nonzero(self):
        pm = PortMapper('/foo[0:5]', np.array([0, 1, 0, 1, 0]))
        self.assertSequenceEqual(pm.get_ports_nonzero(),
                                 [('foo', 1),
                                  ('foo', 3)])

    def test_set_scalar(self):
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        pm['/foo/baz[0:5]'] = 1.0
        assert_array_equal(np.ones(5), pm['/foo/baz[0:5]'])

    def test_set_array(self):
        # Valid empty:
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]')
        new_data = np.arange(10).astype(np.double)
        pm['/foo/bar[0:10]'] = new_data
        assert_array_equal(new_data, pm.data[0:10])

        # Valid nonempty:
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        new_data = np.arange(10).astype(np.double)
        pm['/foo/bar[0:10]'] = new_data
        assert_array_equal(new_data, pm.data[0:10])

    def test_set_discontinuous(self):
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        pm['/foo/*[0:2]'] = 1.0
        np.allclose(np.ones(4), pm['/foo/*[0:2]'])

    def test_get_by_inds(self):
        data = np.random.rand(3)
        pm = PortMapper('/foo[0:3]', data)
        assert_array_equal(data[[0, 1]], pm.get_by_inds([0, 1]))

    def test_set_by_inds(self):
        data = np.random.rand(3)
        pm = PortMapper('/foo[0:3]', data)
        new_data = np.arange(2).astype(np.double)
        pm.set_by_inds([0, 1], new_data)
        assert_array_equal(new_data, pm.get_by_inds([0, 1]))

if __name__ == '__main__':
    main()
