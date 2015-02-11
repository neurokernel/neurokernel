#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal, assert_index_equal, \
    assert_series_equal

from neurokernel.plsel import Selector, SelectorMethods, BasePortMapper, PortMapper

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

class test_selector_class(TestCase):
    def test_selector_add_empty(self):
        s = Selector('')+Selector('')
        assert len(s) == 0
        assert not s.nonempty
        assert s.expanded == ((),)
        assert s.max_levels == 0
        assert s.str == ''

    def test_selector_add_nonempty(self):
        s = Selector('/foo[0]')+Selector('/bar[0]')
        assert len(s) == 2
        assert s.nonempty
        assert s.expanded == (('foo', 0), ('bar', 0))
        assert s.max_levels == 2
        assert s.str == '/foo[0],/bar[0]'

        s = Selector('')+Selector('/foo[0:0]')
        assert len(s) == 0
        assert not s.nonempty
        assert s.expanded == ((),)
        assert s.max_levels == 0
        assert s.str == ''

    def test_selector_concat_empty(self):
        s = Selector.concat(Selector(''), Selector(''))
        assert len(s) == 0
        assert not s.nonempty
        assert s.expanded == ((),)
        assert s.max_levels == 0
        assert s.str == ''

    def test_selector_concat_nonempty(self):
        s = Selector.concat(Selector('[x,y]'), Selector('[0,1]'))
        assert len(s) == 2
        assert s.nonempty
        assert s.expanded == (('x', 0), ('y', 1))
        assert s.max_levels == 2
        assert s.str == '[x,y].+[0,1]'

        self.assertRaises(Exception, Selector.concat, Selector('[x,y]'),
                          Selector('[0:3]'))

    def test_selector_prod_empty(self):
        s = Selector.prod(Selector(''), Selector(''))
        assert len(s) == 0
        assert not s.nonempty
        assert s.expanded == ((),)
        assert s.max_levels == 0
        assert s.str == ''

    def test_selector_prod_nonempty(self):
        s = Selector.prod(Selector('/x'), Selector('[0,1]'))
        assert len(s) == 2
        assert s.nonempty
        assert s.expanded == (('x', 0), ('x', 1))
        assert s.max_levels == 2
        assert s.str == '/x+[0,1]'

        s = Selector.prod(Selector('/x[0:2]'), Selector('[a,b,c]'))
        assert len(s) == 6
        assert s.nonempty
        assert s.expanded == (('x', 0, 'a'), ('x', 0, 'b'), ('x', 0, 'c'),
                              ('x', 1, 'a'), ('x', 1, 'b'), ('x', 1, 'c'))
        assert s.str == '/x[0:2]+[a,b,c]'

    def test_selector_iter(self):
        sel = Selector('/x[0:3]')
        self.assertSequenceEqual([s for s in sel],
                                 [(('x', 0),),
                                  (('x', 1),),
                                  (('x', 2),)])
        sel = Selector('')
        self.assertSequenceEqual([s for s in sel],
                                 [((),)])

class test_path_like_selector(TestCase):
    def setUp(self):
        self.df = df.copy()
        self.sel = SelectorMethods()

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

    def test_select_order(self):
        data = np.random.rand(3)
        df = pd.DataFrame(data,
                          pd.MultiIndex.from_tuples([('foo', i) for i in xrange(3)],
                                                    names=[0, 1]))
        assert_array_equal(self.sel.select(df, '/foo[2,1,0]').values.flatten(),
                           data[[2, 1, 0]])

    def test_are_disjoint(self):
        assert self.sel.are_disjoint('/foo[0:10]/baz',
                                     '/bar[10:20]/qux') == True
        assert self.sel.are_disjoint('/foo[0:10]/baz',
                                     '/foo[5:15]/[baz,qux]') == False

        assert self.sel.are_disjoint('/foo', '') == True
        assert self.sel.are_disjoint('', '') == True
        assert self.sel.are_disjoint('/foo', '/foo', '') == False

        result = self.sel.are_disjoint([['foo', slice(0, 10), 'baz']], 
                                       [['bar', slice(10, 20), 'qux']])
        assert result == True
        result = self.sel.are_disjoint([['foo', slice(0, 10), 'baz']], 
                                       [['foo', slice(5, 15), ['baz','qux']]])
        assert result == False

    def test_count_ports(self):
        result = self.sel.count_ports('/foo/bar[0:2],/moo/[qux,baz]')
        assert result == 4
        result = self.sel.count_ports('')
        assert result == 0

        # XXX Should this be allowed? [] isn't a valid selector:
        result = self.sel.count_ports([])
        assert result == 0

    def test_expand_str(self):
        result = self.sel.expand('/foo/bar[0:2],/moo/[qux,baz]')
        self.assertSequenceEqual(result,
                                 [('foo', 'bar', 0),
                                  ('foo', 'bar', 1),
                                  ('moo', 'qux'),
                                  ('moo', 'baz')])

    def test_expand_list(self):
        result = self.sel.expand([['foo', 'bar', slice(0, 2)],
                                  ['moo', ['qux', 'baz']]])
        self.assertSequenceEqual(result,
                                 [('foo', 'bar', 0),
                                  ('foo', 'bar', 1),
                                  ('moo', 'qux'),
                                  ('moo', 'baz')])

    def test_expand_empty(self):
        self.assertSequenceEqual(self.sel.expand([()]), [()])
        self.assertSequenceEqual(self.sel.expand(''), [()])
        self.assertSequenceEqual(self.sel.expand('/foo[0:0]'), [()])

    def test_expand_pad(self):
        result = self.sel.expand('/foo/bar[0:2],/moo', float('inf'))
        self.assertSequenceEqual(result,
                                 [('foo', 'bar', 0),
                                  ('foo', 'bar', 1),
                                  ('moo', '', '')])
        result = self.sel.expand('/foo/bar[0:2],/moo', 4)
        self.assertSequenceEqual(result,
                                 [('foo', 'bar', 0, ''),
                                  ('foo', 'bar', 1, ''),
                                  ('moo', '', '', '')])

        result = self.sel.expand(Selector('/foo/bar[0:2],/moo'), float('inf'))
        self.assertSequenceEqual(result,
                                 [('foo', 'bar', 0),
                                  ('foo', 'bar', 1),
                                  ('moo', '', '')])
        result = self.sel.expand(Selector('/foo/bar[0:2],/moo'), 4)
        self.assertSequenceEqual(result,
                                 [('foo', 'bar', 0, ''),
                                  ('foo', 'bar', 1, ''),
                                  ('moo', '', '', '')])

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

    def test_get_tuples_empty(self):
        result = self.sel.get_tuples(df, [['foo', 'xxx', 0]])
        self.assertSequenceEqual(result, [])

        result = self.sel.get_tuples(df_single, [['xxx']])
        self.assertSequenceEqual(result, [])

    def test_is_ambiguous_str(self):
        assert self.sel.is_ambiguous('/foo/*') == True
        assert self.sel.is_ambiguous('/foo/[5:]') == True
        assert self.sel.is_ambiguous('/foo/[:10]') == False
        assert self.sel.is_ambiguous('/foo/[5:10]') == False

    def test_is_ambiguous_list(self):
        assert self.sel.is_ambiguous([['foo', '*']]) == True
        assert self.sel.is_ambiguous([['foo', slice(5, None)]]) == True
        assert self.sel.is_ambiguous([['foo', slice(0, 10)]]) == False
        assert self.sel.is_ambiguous([['foo', slice(5, 10)]]) == False

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
        assert self.sel.is_expandable('/foo[0,1,2]') == True
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
                               [[['foo', 'baz'], 'bar', slice(0, 10)]]) == True
        assert self.sel.is_in([['qux', 'bar', [5]]],
                               [[['foo', 'baz'], 'bar', slice(0, 10)]]) == False

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
        assert self.sel.is_selector([('foo', slice(0, 2))]) == True
        assert self.sel.is_selector([('foo', slice(0, None))]) == True
        assert self.sel.is_selector([('foo', [0, 1])]) == True
        assert self.sel.is_selector([('foo', ['a', 'b'])]) == True

        # XXX These are not correct:
        assert self.sel.is_selector([('foo', (0, 1, 2))]) == False
        assert self.sel.is_selector([('foo', 'bar'),
                                     ((0, 1, 2), 0)]) == False
        assert self.sel.is_selector([('foo', ['a', 0])]) == False

    def test_make_index_empty(self):
        idx = self.sel.make_index('')
        assert_index_equal(idx, pd.Index([], dtype='object'))

        idx = self.sel.make_index(Selector(''))
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

    def test_make_index_str_multiple_different_levels(self):
        idx = self.sel.make_index('/foo[0:3],/bar')
        assert_index_equal(idx, pd.MultiIndex(levels=[['bar', 'foo'],
                                                      [0, 1, 2, '']],
                                              labels=[[1, 1, 1, 0],
                                                      [0, 1, 2, 3]]))

    def test_make_index_list_single_level(self):
        idx = self.sel.make_index([['foo']])
        assert_index_equal(idx, pd.Index(['foo'], dtype='object'))
        idx = self.sel.make_index([['foo'], ['bar']])
        assert_index_equal(idx, pd.Index(['foo', 'bar'], dtype='object'))

    def test_make_index_list_multiple_levels(self):
        idx = self.sel.make_index([[['foo', 'bar'], slice(0, 3)]])
        assert_index_equal(idx, pd.MultiIndex(levels=[['bar', 'foo'],
                                                      [0, 1, 2]],
                                              labels=[[1, 1, 1, 0, 0, 0],
                                                      [0, 1, 2, 0, 1, 2]]))

    def test_make_index_list_multiple_different_levels(self):
        idx = self.sel.make_index([['foo', [0, 1, 2]], ['bar']])
        assert_index_equal(idx, pd.MultiIndex(levels=[['bar', 'foo'],
                                                      [0, 1, 2, '']],
                                              labels=[[1, 1, 1, 0],
                                                      [0, 1, 2, 3]]))

    def test_make_index_invalid(self):
        self.assertRaises(Exception, self.sel.make_index, 'foo/bar[')

    def test_max_levels_str(self):
        assert self.sel.max_levels('/foo/bar[0:10]') == 3
        assert self.sel.max_levels('/foo/bar[0:10],/baz/qux') == 3

    def test_max_levels_list(self):
        assert self.sel.max_levels([['foo', 'bar', slice(0, 10)]]) == 3
        assert self.sel.max_levels([['foo', 'bar', slice(0, 10)],
                                    ['baz', 'qux']]) == 3

    def test_pad_parsed(self):
        sel = [['x', 'y'], ['a', 'b', 'c']]
        sel_id = id(sel)
        sel_padded = self.sel.pad_parsed(sel, float('inf'))
        self.assertSequenceEqual(sel_padded,
                                 [['x', 'y', ''], ['a', 'b', 'c']])
        assert sel_id == id(sel_padded)

        sel = [['x', 'y'], ['a', 'b', 'c']]
        sel_id = id(sel)
        sel_padded = self.sel.pad_parsed(sel, 4)
        self.assertSequenceEqual(sel_padded,
                                 [['x', 'y', '', ''], ['a', 'b', 'c', '']])
        assert sel_id == id(sel_padded)
        
        sel = [['x', 'y'], ['a', 'b', 'c']]
        sel_id = id(sel)
        sel_padded = self.sel.pad_parsed(sel, float('inf'), False)
        self.assertSequenceEqual(sel_padded,
                                 [['x', 'y', ''], ['a', 'b', 'c']])
        assert sel_id != id(sel_padded)

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
        pm0 = BasePortMapper('/foo[0:5],/bar[0:5]', range(10))
        pm1 = BasePortMapper('/bar[0:5],/foo[0:5]', range(5, 10)+range(5))
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
        pm0 = BasePortMapper('/foo[0:5],/bar[0:5]', range(5)*2)
        pm1 = BasePortMapper.from_index(pm0.index, range(5)*2)
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
        pm = BasePortMapper('/foo[0:5],/bar[0:5]', range(10, 20))
        self.assertSequenceEqual(pm.inds_to_ports([14, 15]),
                                 [('foo', 4), ('bar', 0)])

    def test_ports_to_inds(self):
        # Without a specified port map:
        pm = BasePortMapper('/foo[0:5],/bar[0:5]')
        np.allclose(pm.ports_to_inds('/foo[4],/bar[0]'), [4, 5])

        # With a specified port map:
        pm = BasePortMapper('/foo[0:5],/bar[0:5]', range(10, 20))
        np.allclose(pm.ports_to_inds('/foo[4],/bar[0]'), [14, 15])

    def test_get_map(self):
        # Try to get selector that is in the mapper:
        pm = BasePortMapper('/foo[0:5],/bar[0:5]')
        self.assertSequenceEqual(pm.get_map('/bar[0:5]').tolist(), range(5, 10))

        # Try to get selector that is not in the mapper:
        self.assertSequenceEqual(pm.get_map('/foo[5:10]').tolist(), [])

    def test_set_map(self):
        pm = BasePortMapper('/foo[0:5],/bar[0:5]')
        pm.set_map('/bar[0:5]', range(5))
        self.assertSequenceEqual(pm.portmap.ix[5:10].tolist(), range(5))

class test_port_mapper(TestCase):
    def setUp(self):
        self.data = np.random.rand(20)

    def test_create(self):

        # Empty selector, empty data:
        pm = PortMapper('')
        assert_series_equal(pm.portmap, pd.Series([], dtype=np.int64))
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
        self.assertRaises(Exception, PortMapper, '', [1, 2, 3])

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

    def test_set(self):
        # Mapper with data:
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)
        pm['/foo/baz[0:5]'] = 1.0
        np.allclose(np.ones(5), pm['/foo/baz[0:5]'])
        
        # Mapper without data:
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]')
        self.assertRaises(Exception, pm.__setitem__, '/foo/bar[0]', 0)

    def test_set_discontinuous(self):
        pm = PortMapper('/foo/bar[0:10],/foo/baz[0:10]', self.data)                        
        pm['/foo/*[0:2]'] = 1.0
        np.allclose(np.ones(4), pm['/foo/*[0:2]'])

if __name__ == '__main__':
    main()
