#!/usr/bin/env python

"""
XPATH-like row selector for pandas DataFrames with hierarchical MultiIndexes.
"""

import re
import numpy as np

import pandas as pd
import ply.lex as lex

class XPathSelector(object):
    """
    Class for selecting rows of a pandas DataFrame using XPATH-like selectors. 

    Select rows from a pandas DataFrame using XPATH-like selectors.
    Assumes that the DataFrame instance has a MultiIndex where each level
    corresponds to a level of the selector; a level may either be a denoted by a
    string label (e.g., 'foo') or a numerical index (e.g., 0, 1, 2). 
    Examples of valid selectors include

    /foo
    /foo/bar
    /foo/bar[0]
    /foo/bar[0:5]
    /foo/*/baz
    /foo/*/baz[5]

    Notes
    -----
    Numerical indices are assumed to be zero-based.
    """

    tokens = ('ASTERISK', 'INTEGER', 'INTERVAL', 'STRING')

    def __init__(self):
        self._build()

    def _parse_interval_str(self, s):
        """
        Convert string representation of interval to tuple containing numerical
        start and stop values.
        """

        start, stop = s.split(':')
        if start == '':
            start = 0
        else:
            start = int(start)
        if stop == '':
            stop = np.inf
        else:
            stop = int(stop)
        return (start, stop)

    def t_ASTERISK(self, t):
        r'/\*'
        t.value = t.value.strip('/')
        return t

    def t_INTEGER(self, t):
        r'\[\d+\]'
        t.value = int(re.search('\[(\d+)\]', t.value).group(1))
        return t

    def t_INTERVAL(self, t):
        r'\[\d*\:\d*\]'
        t.value = self._parse_interval_str(re.search('\[(.+)\]', t.value).group(1))
        return t

    def t_STRING(self, t):
        r'/[^*/\[\]:]+'
        t.value = t.value.strip('/')
        return t
            
    def t_error(self, t):
        print 'Illegal character "%s"' % t.value[0]
        raise ValueError('Cannot parse selector')

    def _build(self, **kwargs):
        """
        Build lexer.
        """

        self.lexer = lex.lex(module=self, **kwargs)

    def parse(self, selector):
        """
        Parse a specified selector string into tokens.
        """

        self.lexer.input(selector)
        token_list = []
        while True:
            token = self.lexer.token()
            if not token: break
            token_list.append(token)
        return token_list

    def _select_test(self, row, token_list, start=None, stop=None):
        """
        Method for checking whether the entries in a subinterval of a tuple of
        data corresponding to the entries of one row in a DataFrame match the
        specified token values.
        """

        row_sub = row[start:stop]
        for i, token in enumerate(token_list):
            if token.type == 'ASTERISK':
                continue
            elif token.type in ['INTEGER', 'STRING']:
                if row_sub[i] != token.value:
                    return False
            elif token.type == 'INTERVAL':
                i_start, i_stop = token.value
                if not(row_sub[i] >= i_start and row_sub[i] < i_stop):
                    return False
            else:
                continue
        return True
        
    def get_index(self, df, selector, start=None, stop=None):
        """
        Return MultiIndex corresponding to rows selected by specified selector.
        """

        token_list = self.parse(selector)

        # The number of tokens must not exceed the number of levels in the
        # DataFrame's MultiIndex:        
        if len(token_list) > len(df.index.names[start:stop]):
            raise ValueError('Number of levels in selector exceeds that of '
                             'DataFrame index')
            
        # XXX This probably could be made faster by directly manipulating the
        # existing MultiIndex:
        return pd.MultiIndex.from_tuples([t for t in df.index if \
                                          self._select_test(t, token_list,
                                                            start, stop)])
        
    def select(self, df, selector, start=None, stop=None):
        """
        Select rows from DataFrame.
        """

        token_list = self.parse(selector)

        # The number of tokens must not exceed the number of levels in the
        # DataFrame's MultiIndex:        
        if len(token_list) > len(df.index.names[start:stop]):
            raise ValueError('Number of levels in selector exceeds number in row subinterval')
            
        return df.select(lambda row: self._select_test(row, token_list, start, stop))

    def _isvalidvarname(self, s):
        """
        Return True if the given string is a valid Python variable identifier.

        Notes
        -----
        A valid Python variable identifier must start with an alphabetical character or '_'
        followed by alphanumeric characters.
        """

        try:
            result = re.match('[a-zA-Z_]\w*', s)
        except TypeError:
            return False
        else:
            if result:
                return True
            else:
                return False

    def query(self, df, selector):
        """
        Select rows from DataFrame.

        Notes
        -----
        Seems slower than the select() method.
        """

        token_list = self.parse(selector)

        if len(token_list) > len(df.index.names):
            raise ValueError('Number of levels in selector exceeds that of '
                             'DataFrame index')

        # This method can only work if the MultiIndex level names can be valid
        # Python variable identifiers:
        assert all(map(self._isvalidvarname, df.index.names))

        expr_list = []
        for i, token in enumerate(token_list):
            if token.type == 'ASTERISK':
                expr_list.append(df.index.names[i] + ' == ' + df.index.names[i])
            elif token.type == 'INTEGER':
                expr_list.append(df.index.names[i] + ' == %i' % token.value)
            elif token.type == 'STRING':
                expr_list.append(df.index.names[i] + ' == \'%s\'' % token.value)
            elif token.type == 'INTERVAL':
                expr_list.append(df.index.names[i] + ' >= %i' % token.value[0])
                if not np.isinf(token.value[1]):
                    expr_list.append(df.index.names[i] + ' < %i' % token.value[1])
            else:
                continue
        return df.query(' and '.join(expr_list))

df = pd.DataFrame(data={'data': np.random.rand(12),
                        'level_0': ['foo', 'foo', 'foo', 'foo', 'foo', 'foo',
                                    'bar', 'bar', 'bar', 'bar', 'baz', 'baz'],                        
                        'level_1': ['qux', 'qux', 'qux', 'qux', 'mof', 'mof',
                                    'qux', 'qux', 'qux', 'mof', 'mof', 'mof'],
                        'level_2': ['xxx', 'yyy', 'yyy', 'yyy', 'zzz', 'zzz',
                                    'xxx', 'xxx', 'yyy', 'zzz', 'yyy', 'zzz'],
                        'level_3': [0, 0, 1, 2, 0, 1, 
                                    0, 1, 0, 1, 0, 1]})
df.set_index('level_0', append=False, inplace=True)
df.set_index('level_1', append=True, inplace=True)
df.set_index('level_2', append=True, inplace=True)
df.set_index('level_3', append=True, inplace=True)

if __name__ == '__main__':
    from unittest import main, TestCase
    from pandas.util.testing import assert_frame_equal

    class test_xpath_selector(TestCase):
        def setUp(self):
            self.df = pd.DataFrame(data={'data': np.random.rand(10),
                                         0: ['foo', 'foo', 'foo', 'foo', 'foo',
                                             'bar', 'bar', 'bar',
                                             'baz', 'baz'],
                                         1: ['qux', 'qux', 'mof', 'mof', 'mof',
                                             'qux', 'qux', 'qux', 'qux', 'mof'],
                                         2: [0, 1, 0, 1, 2, 0, 1, 2, 0, 0]})
            self.df.set_index(0, append=False, inplace=True)
            self.df.set_index(1, append=True, inplace=True)
            self.df.set_index(2, append=True, inplace=True)
            self.sel = XPathSelector()
        def test_str_one(self):
            result = self.sel.select(self.df, '/foo')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_asterisk(self):
            result = self.sel.select(self.df, '/*/qux')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('bar','qux',0),
                                             ('bar','qux',1),
                                             ('bar','qux',2),
                                             ('baz','qux',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_integer(self):
            result = self.sel.select(self.df, '/bar/qux[1]')
            idx = pd.MultiIndex.from_tuples([('bar','qux',1)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_interval_0(self):
            result = self.sel.select(self.df, '/foo/mof[:]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_interval_1(self):
            result = self.sel.select(self.df, '/foo/mof[1:]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',1),
                                             ('foo','mof',2)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_interval_2(self):
            result = self.sel.select(self.df, '/foo/mof[:2]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',0),
                                             ('foo','mof',1)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_interval_3(self):
            result = self.sel.select(self.df, '/bar/qux[0:2]')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('bar','qux',1)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

    main()

