#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_index_equal

from neurokernel.plsel import PathLikeSelector, PortMapper

df1 = pd.DataFrame(data={'data': np.random.rand(12),
                   'level_0': ['foo', 'foo', 'foo', 'foo', 'foo', 'foo',
                               'bar', 'bar', 'bar', 'bar', 'baz', 'baz'],
                   'level_1': ['qux', 'qux', 'qux', 'qux', 'mof', 'mof',
                               'qux', 'qux', 'qux', 'mof', 'mof', 'mof'],
                   'level_2': ['xxx', 'yyy', 'yyy', 'yyy', 'zzz', 'zzz',
                               'xxx', 'xxx', 'yyy', 'zzz', 'yyy', 'zzz'],
                   'level_3': [0, 0, 1, 2, 0, 1,
                               0, 1, 0, 1, 0, 1]})
df1.set_index('level_0', append=False, inplace=True)
df1.set_index('level_1', append=True, inplace=True)
df1.set_index('level_2', append=True, inplace=True)
df1.set_index('level_3', append=True, inplace=True)

df = pd.DataFrame(data={'data': np.random.rand(10),
                  0: ['foo', 'foo', 'foo', 'foo', 'foo',
                      'bar', 'bar', 'bar', 'baz', 'baz'],
                  1: ['qux', 'qux', 'mof', 'mof', 'mof',
                      'qux', 'qux', 'qux', 'qux', 'mof'],
                  2: [0, 1, 0, 1, 2, 
                      0, 1, 2, 0, 0]})
df.set_index(0, append=False, inplace=True)
df.set_index(1, append=True, inplace=True)
df.set_index(2, append=True, inplace=True)

df2 = pd.DataFrame(data={'data': np.random.rand(10),
                  0: ['foo', 'foo', 'foo', 'foo', 'foo',
                      'bar', 'bar', 'bar', 'baz', 'baz'],
                  1: ['qux', 'qux', 'mof', 'mof', 'mof',
                      'qux', 'qux', 'qux', 'qux', 'mof'],
                  2: [0, 1, 0, 1, 2, 
                      0, 1, 2, 0, 0]})
df2.set_index(0, append=False, inplace=True)
df2.set_index(1, append=True, inplace=True)
df2.set_index(2, append=True, inplace=True)

df_single = pd.DataFrame(data={'data': np.random.rand(5),
                               0: ['foo', 'foo', 'bar', 'bar', 'baz']})
df_single.set_index(0, append=False, inplace=True)

class test_path_like_selector(TestCase):
    def setUp(self):
        self.df = df.copy()
        self.sel = PathLikeSelector()

    def test_select_empty(self):
        result = self.sel.select(self.df, [[]])
        assert_frame_equal(result, self.df.drop(self.df.index))

    def test_select_str(self):
        result = self.sel.select(self.df, '/foo')
        idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                         ('foo','qux',1),
                                         ('foo','mof',0),
                                         ('foo','mof',1),
                                         ('foo','mof',2)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_list(self):
        result = self.sel.select(self.df, [['foo']])
        idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                         ('foo','qux',1),
                                         ('foo','mof',0),
                                         ('foo','mof',1),
                                         ('foo','mof',2)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_comma(self):
        result = self.sel.select(self.df, '/foo/qux,/baz/mof')
        idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                         ('foo','qux',1),
                                         ('baz','mof',0)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_plus(self):
        result = self.sel.select(self.df, '/foo+/qux+[0,1]')
        idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                         ('foo','qux',1)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_dotplus(self):
        result = self.sel.select(self.df, '/[bar,baz].+/[qux,mof].+/[0,0]')
        idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                         ('baz','mof',0)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_paren(self):
        result = self.sel.select(self.df, '(/bar,/baz)')
        idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                         ('bar','qux',1),
                                         ('bar','qux',2),
                                         ('baz','qux',0),
                                         ('baz','mof',0)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_paren_plus(self):
        result = self.sel.select(self.df, '(/bar,/baz)+/qux')
        idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                         ('bar','qux',1),
                                         ('bar','qux',2),
                                         ('baz','qux',0)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_asterisk(self):
        result = self.sel.select(self.df, '/*/qux')
        idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                         ('foo','qux',1),
                                         ('bar','qux',0),
                                         ('bar','qux',1),
                                         ('bar','qux',2),
                                         ('baz','qux',0)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_integer_with_brackets(self):
        result = self.sel.select(self.df, '/bar/qux[1]')
        idx = pd.MultiIndex.from_tuples([('bar','qux',1)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_integer_no_brackets(self):
        result = self.sel.select(self.df, '/bar/qux/1')
        idx = pd.MultiIndex.from_tuples([('bar','qux',1)], names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_integer_set(self):
        result = self.sel.select(self.df, '/foo/qux[0,1]')
        idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                         ('foo','qux',1)],
                                        names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_string_set(self):
        result = self.sel.select(self.df, '/foo/[qux,mof]')
        idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                         ('foo','qux',1),
                                         ('foo','mof',0),
                                         ('foo','mof',1),
                                         ('foo','mof',2)],
                                        names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_interval_no_bounds(self):
        result = self.sel.select(self.df, '/foo/mof[:]')
        idx = pd.MultiIndex.from_tuples([('foo','mof',0),
                                         ('foo','mof',1),
                                         ('foo','mof',2)],
                                        names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_interval_lower_bound(self):
        result = self.sel.select(self.df, '/foo/mof[1:]')
        idx = pd.MultiIndex.from_tuples([('foo','mof',1),
                                         ('foo','mof',2)],
                                        names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_interval_upper_bound(self):
        result = self.sel.select(self.df, '/foo/mof[:2]')
        idx = pd.MultiIndex.from_tuples([('foo','mof',0),
                                         ('foo','mof',1)],
                                        names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_select_interval_both_bounds(self):
        result = self.sel.select(self.df, '/bar/qux[0:2]')
        idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                         ('bar','qux',1)],
                                        names=[0, 1, 2])
        assert_frame_equal(result, self.df.ix[idx])

    def test_are_disjoint_str(self):
        assert self.sel.are_disjoint('/foo[0:10]/baz',
                                     '/bar[10:20]/qux') == True
        assert self.sel.are_disjoint('/foo[0:10]/baz',
                                     '/foo[5:15]/[baz,qux]') == False

    def test_are_disjoint_list(self):
        result = self.sel.are_disjoint([['foo', (0, 10), 'baz']], 
                                       [['bar', (10, 20), 'qux']])
        assert result == True
        result = self.sel.are_disjoint([['foo', (0, 10), 'baz']], 
                                       [['foo', (5, 15), ['baz','qux']]])
        assert result == False

    def test_count_ports(self):
        result = self.sel.count_ports('/foo/bar[0:2],/moo/[qux,baz]')
        assert result == 4

    def test_expand_str(self):
        result = self.sel.expand('/foo/bar[0:2],/moo/[qux,baz]')
        self.assertSequenceEqual(result,
                                 [('foo', 'bar', 0),
                                  ('foo', 'bar', 1),
                                  ('moo', 'qux'), 
                                  ('moo', 'baz')])

    def test_expand_list(self):
        result = self.sel.expand([['foo', 'bar', (0, 2)],
                                  ['moo', ['qux', 'baz']]])
        self.assertSequenceEqual(result,
                                 [('foo', 'bar', 0),
                                  ('foo', 'bar', 1),
                                  ('moo', 'qux'), 
                                  ('moo', 'baz')])

    def test_get_index_str(self):
        idx = self.sel.get_index(self.df, '/foo/mof/*')
        assert_index_equal(idx, pd.MultiIndex(levels=[['foo'], ['mof'],
                                                      [0, 1, 2]],
                                              labels=[[0, 0, 0],
                                                      [0, 0, 0],
                                                      [0, 1, 2]]))

    def test_get_index_list(self):
        idx = self.sel.get_index(self.df, [['foo', 'mof', '*']])
        assert_index_equal(idx, pd.MultiIndex(levels=[['foo'], ['mof'],
                                                      [0, 1, 2]],
                                              labels=[[0, 0, 0],
                                                      [0, 0, 0],
                                                      [0, 1, 2]]))

    def test_get_tuples_str(self):
        result = self.sel.get_tuples(df, '/foo/mof/*')
        self.assertSequenceEqual(result,
                                 [('foo', 'mof', 0),
                                  ('foo', 'mof', 1),
                                  ('foo', 'mof', 2)])

        result = self.sel.get_tuples(df_single, '/foo')
        self.assertSequenceEqual(result,
                                 [('foo',),
                                  ('foo',)])

    def test_get_tuples_list(self):
        result = self.sel.get_tuples(df, [['foo', 'mof', '*']])
        self.assertSequenceEqual(result,
                                 [('foo', 'mof', 0),
                                  ('foo', 'mof', 1),
                                  ('foo', 'mof', 2)])

        result = self.sel.get_tuples(df_single, [['foo']])
        self.assertSequenceEqual(result,
                                 [('foo',),
                                  ('foo',)])

    def test_is_ambiguous_str(self):
        assert self.sel.is_ambiguous('/foo/*') == True
        assert self.sel.is_ambiguous('/foo/[5:]') == True
        assert self.sel.is_ambiguous('/foo/[:10]') == False
        assert self.sel.is_ambiguous('/foo/[5:10]') == False

    def test_is_ambiguous_list(self):
        assert self.sel.is_ambiguous([['foo', '*']]) == True
        assert self.sel.is_ambiguous([['foo', (5, np.inf)]]) == True
        assert self.sel.is_ambiguous([['foo', (0, 10)]]) == False
        assert self.sel.is_ambiguous([['foo', (5, 10)]]) == False

    def test_is_identifier(self):
        assert self.sel.is_identifier('/foo/bar') == True
        assert self.sel.is_identifier(0) == False
        assert self.sel.is_identifier('foo') == False
        #assert self.sel.is_identifier('0') == False # this doesn't work
        assert self.sel.is_identifier(['foo', 'bar']) == True
        assert self.sel.is_identifier(['foo', 0]) == True
        assert self.sel.is_identifier(['foo', [0, 1]]) == False
        assert self.sel.is_identifier([['foo', 'bar']]) == True
        assert self.sel.is_identifier([['foo', 'bar'], ['baz']]) == False
        assert self.sel.is_identifier([['foo', 0]]) == True

    def test_to_identifier(self):
        assert self.sel.to_identifier(['foo']) == '/foo'
        assert self.sel.to_identifier(['foo', 0]) == '/foo[0]'
        self.assertRaises(Exception, self.sel.to_identifier, 'foo')
        self.assertRaises(Exception, self.sel.to_identifier, 
                          [['foo', ['a', 'b']]])
        self.assertRaises(Exception, self.sel.to_identifier, 
                          ['foo', (0, 2)])

    def test_index_to_selector(self):
        idx = self.sel.make_index('/foo,/bar')
        self.assertSequenceEqual(self.sel.index_to_selector(idx),
                                 [('foo',), ('bar',)])
        idx = self.sel.make_index('/foo[0:2]')
        self.assertSequenceEqual(self.sel.index_to_selector(idx),
                                 [('foo', 0), ('foo', 1)])

    def test_is_expandable(self):
        assert self.sel.is_expandable('') == False

        assert self.sel.is_expandable('/foo') == False
        assert self.sel.is_expandable('/foo/bar') == False
        assert self.sel.is_expandable('/foo/*') == False

        assert self.sel.is_expandable([['foo']]) == False
        assert self.sel.is_expandable([['foo', 'bar']]) == False

        assert self.sel.is_expandable('/foo[0:2]') == True
        assert self.sel.is_expandable('[0:2]') == True

        assert self.sel.is_expandable([['foo', [0, 1]]]) == True
        assert self.sel.is_expandable([['foo', 0],
                                       ['foo', 1]]) == True
        assert self.sel.is_expandable([[[0, 1]]]) == True

    def test_is_in_str(self):
        assert self.sel.is_in('', '/foo[0:5]') == True
        assert self.sel.is_in('/foo/bar[5]', '/[foo,baz]/bar[0:10]') == True
        assert self.sel.is_in('/qux/bar[5]', '/[foo,baz]/bar[0:10]') == False

    def test_is_in_list(self):
        assert self.sel.is_in([()], [('foo', 0), ('foo', 1)])
        assert self.sel.is_in([['foo', 'bar', [5]]],
                               [[['foo', 'baz'], 'bar', (0, 10)]]) == True
        assert self.sel.is_in([['qux', 'bar', [5]]],
                               [[['foo', 'baz'], 'bar', (0, 10)]]) == False

    def test_is_selector_empty(self):
        assert self.sel.is_selector_empty('') == True            
        assert self.sel.is_selector_empty([[]]) == True
        assert self.sel.is_selector_empty([()]) == True
        assert self.sel.is_selector_empty(((),)) == True
        assert self.sel.is_selector_empty([[], []]) == True
        assert self.sel.is_selector_empty([(), []]) == True
        assert self.sel.is_selector_empty(((), [])) == True

        assert self.sel.is_selector_empty('/foo') == False
        assert self.sel.is_selector_empty('/foo/*') == False
        assert self.sel.is_selector_empty([['foo']]) == False
        assert self.sel.is_selector_empty([['foo', 'bar']]) == False
        assert self.sel.is_selector_empty([['']]) == False # is this correct?

    def test_is_selector_str(self):
        assert self.sel.is_selector('') == True
        assert self.sel.is_selector('/foo') == True
        assert self.sel.is_selector('/foo/bar') == True
        assert self.sel.is_selector('/foo!?') == True
        assert self.sel.is_selector('/foo[0]') == True
        assert self.sel.is_selector('/foo[0:2]') == True
        assert self.sel.is_selector('/foo[0:]') == True
        assert self.sel.is_selector('/foo[:2]') == True
        assert self.sel.is_selector('/foo/*') == True
        assert self.sel.is_selector('/foo,/bar') == True
        assert self.sel.is_selector('/foo+/bar') == True
        assert self.sel.is_selector('/foo[0:2].+/bar[0:2]') == True

        assert self.sel.is_selector('/foo[') == False
        assert self.sel.is_selector('foo[0]') == False

    def test_is_selector_list(self):
        assert self.sel.is_selector([[]]) == True
        assert self.sel.is_selector([['foo', 'bar']]) == True
        assert self.sel.is_selector([('foo', 'bar')]) == True
        assert self.sel.is_selector([('foo', '*')]) == True
        assert self.sel.is_selector([('foo', 'bar'), ('bar', 'qux')]) == True
        assert self.sel.is_selector([('foo', 0)]) == True
        assert self.sel.is_selector([('foo', (0, 2))]) == True
        assert self.sel.is_selector([('foo', (0, np.inf))]) == True
        assert self.sel.is_selector([('foo', [0, 1])]) == True
        assert self.sel.is_selector([('foo', ['a', 'b'])]) == True

        assert self.sel.is_selector([('foo', (0, 1, 2))]) == False
        assert self.sel.is_selector([('foo', 'bar'),
                                     ((0, 1, 2), 0)]) == False
        assert self.sel.is_selector([('foo', ['a', 0])]) == False

    def test_make_index_empty(self):
        idx = self.sel.make_index('')
        assert_index_equal(idx, pd.Index([], dtype='object'))

    def test_make_index_str_single_level(self):
        idx = self.sel.make_index('/foo')
        assert_index_equal(idx, pd.Index(['foo'], dtype='object'))
        idx = self.sel.make_index('/foo,/bar')
        assert_index_equal(idx, pd.Index(['foo', 'bar'], dtype='object'))

    def test_make_index_str_multiple_levels(self):
        idx = self.sel.make_index('/[foo,bar]/[0:3]')
        assert_index_equal(idx, pd.MultiIndex(levels=[['bar', 'foo'],
                                                      [0, 1, 2]],
                                              labels=[[1, 1, 1, 0, 0, 0],
                                                      [0, 1, 2, 0, 1, 2]]))

    def test_make_index_list_single_level(self):
        idx = self.sel.make_index([['foo']])
        assert_index_equal(idx, pd.Index(['foo'], dtype='object'))
        idx = self.sel.make_index([['foo'], ['bar']])
        assert_index_equal(idx, pd.Index(['foo', 'bar'], dtype='object'))

    def test_make_index_list_multiple_levels(self):
        idx = self.sel.make_index([[['foo', 'bar'], (0, 3)]])
        assert_index_equal(idx, pd.MultiIndex(levels=[['bar', 'foo'],
                                                      [0, 1, 2]],
                                              labels=[[1, 1, 1, 0, 0, 0],
                                                      [0, 1, 2, 0, 1, 2]]))

    def test_make_index_invalid(self):
        self.assertRaises(Exception, self.sel.make_index, 'foo/bar[')

    def test_max_levels_str(self):
        assert self.sel.max_levels('/foo/bar[0:10]') == 3
        assert self.sel.max_levels('/foo/bar[0:10],/baz/qux') == 3

    def test_max_levels_list(self):
        assert self.sel.max_levels([['foo', 'bar', (0, 10)]]) == 3
        assert self.sel.max_levels([['foo', 'bar', (0, 10)],
                                    ['baz', 'qux']]) == 3

class test_port_mapper(TestCase):
    def setUp(self):
        self.data = np.random.rand(20)

    def test_get(self):
        pm = PortMapper(self.data,
                        '/foo/bar[0:10],/foo/baz[0:10]')
        np.allclose(self.data[0:10], pm['/foo/bar[0:10]'])

    def test_get_discontinuous(self):
        pm = PortMapper(self.data,
                        '/foo/bar[0:10],/foo/baz[0:10]')
        np.allclose(self.data[[0, 2, 4, 6]],
                    pm['/foo/bar[0,2,4,6]'])

    def test_get_sub(self):
        pm = PortMapper(self.data,
                        '/foo/bar[0:5],/foo/baz[0:5]',
                        np.arange(5, 15))
        np.allclose(self.data[5:10], pm['/foo/bar[0:5]'])

    def test_get_ports(self):
        pm = PortMapper(np.arange(10), '/foo/bar[0:10]')
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
        pm = PortMapper(np.array([0, 1, 0, 1, 0]), '/foo[0:5]')
        np.allclose(pm.get_ports_as_inds(lambda x: np.asarray(x, dtype=np.bool)), 
                    [1, 3])

    def test_get_ports_nonzero(self):
        pm = PortMapper(np.array([0, 1, 0, 1, 0]), '/foo[0:5]')
        self.assertSequenceEqual(pm.get_ports_nonzero(),
                                 [('foo', 1),
                                  ('foo', 3)])

    def test_inds_to_ports(self):
        pm = PortMapper(np.random.rand(10), '/foo[0:5],/bar[0:5]')
        self.assertSequenceEqual(pm.inds_to_ports([4, 5]),
                                 [('foo', 4), ('bar', 0)])

    def test_ports_to_inds(self):
        pm = PortMapper(np.random.rand(10), '/foo[0:5],/bar[0:5]')
        np.allclose(pm.ports_to_inds('/foo[4],/bar[0]'), [4, 5])

    def test_set(self):
        pm = PortMapper(self.data,
                        '/foo/bar[0:10],/foo/baz[0:10]')
        pm['/foo/baz[0:5]'] = 1.0
        np.allclose(np.ones(5), pm['/foo/baz[0:5]'])

    def test_set_discontinuous(self):
        pm = PortMapper(self.data,
                        '/foo/bar[0:10],/foo/baz[0:10]')
        pm['/foo/*[0:2]'] = 1.0
        np.allclose(np.ones(4), pm['/foo/*[0:2]'])

if __name__ == '__main__':
    main()
