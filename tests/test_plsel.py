#!/usr/bin/env python

from unittest import main, TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal, assert_index_equal, \
    assert_series_equal

from neurokernel.plsel import Selector, SelectorMethods

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
        assert s.str == '/foo/0,/bar/0'

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
        assert s.str == '/x/0,/y/1'

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
        assert s.str == '/x/0,/x/1'

        s = Selector.prod(Selector('/x[0:2]'), Selector('[a,b,c]'))
        assert len(s) == 6
        assert s.nonempty
        assert s.expanded == (('x', 0, 'a'), ('x', 0, 'b'), ('x', 0, 'c'),
                              ('x', 1, 'a'), ('x', 1, 'b'), ('x', 1, 'c'))
        assert s.str == '/x/0/a,/x/0/b,/x/0/c,/x/1/a,/x/1/b,/x/1/c'

    def test_selector_iter(self):
        sel = Selector('/x[0:3]')
        self.assertSequenceEqual([s for s in sel],
                                 [(('x', 0),),
                                  (('x', 1),),
                                  (('x', 2),)])
        sel = Selector('')
        self.assertSequenceEqual([s for s in sel],
                                 [((),)])

    def test_selector_union_empty(self):
        a = Selector('')
        b = Selector('')
        c = Selector.union(a, b)
        assert len(c) == 0
        assert c.expanded == ((),)
        assert c.max_levels == 0
        assert c.str == ''

    def test_selector_union_nonempty(self):
        a = Selector('/x[0:3]')
        b = Selector('/x[2:5]')
        c = Selector.union(a, b)
        assert len(c) == 5
        assert c.expanded == (('x', 0), ('x', 1), ('x', 2), ('x', 3), ('x', 4))
        assert c.max_levels == 2
        assert c.str == '/x/0,/x/1,/x/2,/x/3,/x/4'

    def test_selector_union_empty_nonempty(self):
        a = Selector('')
        b = Selector('/x[0:3]')
        c = Selector.union(a, b)
        assert len(c) == 3
        assert c.expanded == (('x', 0), ('x', 1), ('x', 2))
        assert c.max_levels == 2
        assert c.str == '/x/0,/x/1,/x/2'

    def test_selector_identifiers(self):
        a = Selector('/x[0:3]')
        assert a.identifiers == ['/x/0', '/x/1', '/x/2']

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

    def test_pad_tuple_list(self):
        x = [('a', 'b'), ('c', 'd')]
        self.assertSequenceEqual(self.sel.pad_tuple_list(x, 0), x)
        self.assertSequenceEqual(self.sel.pad_tuple_list(x, 3),
                                 [('a', 'b', ''), ('c', 'd', '')])
        x = [('a', 'b'), ('c',)]
        self.assertSequenceEqual(self.sel.pad_tuple_list(x, 0), x)
        self.assertSequenceEqual(self.sel.pad_tuple_list(x, 3),
                                 [('a', 'b', ''), ('c', '', '')])

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

    def test_tokens_to_str(self):
        assert self.sel.tokens_to_str([]) == ''
        assert self.sel.tokens_to_str(['a']) == '/a'
        assert self.sel.tokens_to_str(['a', 0]) == '/a/0'
        assert self.sel.tokens_to_str(('a', 0)) == '/a/0'
        assert self.sel.tokens_to_str(['a', '*']) == '/a/*'
        assert self.sel.tokens_to_str(['a', 'b', 0]) == '/a/b/0'
        assert self.sel.tokens_to_str(['a', 'b', [0, 1]]) == '/a/b[0,1]'
        assert self.sel.tokens_to_str(['a', 'b', (0, 1)]) == '/a/b[0,1]'
        assert self.sel.tokens_to_str(['a', 'b', slice(0, 5)]) == '/a/b[0:5]'
        assert self.sel.tokens_to_str(['a', 'b', slice(None, 5)]) == '/a/b[:5]'

    def test_collapse(self):
        assert self.sel.collapse([]) == ''
        assert self.sel.collapse([['a']]) == '/a'
        assert self.sel.collapse([['a', 0]]) == '/a/0'
        assert self.sel.collapse([('a', 0)]) == '/a/0'
        assert self.sel.collapse([['a', 'b', 0]]) == '/a/b/0'
        assert self.sel.collapse([['a', 0], ['b', 0]]) == '/a/0,/b/0'
        assert self.sel.collapse([['a', 'b', (0, 1)], ['c', 'd']]) == '/a/b[0,1],/c/d'
        
if __name__ == '__main__':
    main()
