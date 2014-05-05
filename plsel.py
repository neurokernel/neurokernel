#!/usr/bin/env python

"""
Path-like row selector for pandas DataFrames with hierarchical MultiIndexes.
"""

import re
import numpy as np

import pandas as pd
import ply.lex as lex
import ply.yacc as yacc

class PathLikeSelector(object):
    """
    Class for selecting rows of a pandas DataFrame using path-like selectors.

    Select rows from a pandas DataFrame using path-like selectors.
    Assumes that the DataFrame instance has a MultiIndex where each level
    corresponds to a level of the selector; a level may either be a denoted by a
    string label (e.g., 'foo') or a numerical index (e.g., 0, 1, 2).
    Examples of valid selectors include

    /foo/bar
    /foo/[qux,bar]
    /foo/bar[0]
    /foo/bar[0,1]
    /foo/bar[0:5]
    /foo/*/baz
    /foo/*/baz[5]
    /foo/bar+/baz/qux

    The class can also be used to create new MultiIndex instances from selectors
    that contain no wildcards.

    Notes
    -----
    Numerical indices are assumed to be zero-based. Ranges do not include the
    end element (i.e., like numpy, not like pandas).
    """

    tokens = ('ASTERISK', 'INTEGER', 'INTEGER_SET',
              'INTERVAL', 'STRING', 'STRING_SET', 'PLUS')

    def __init__(self):
        self._setup()

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

    def t_PLUS(self, t):
        r'\+'
        return t

    def t_ASTERISK(self, t):
        r'/\*'
        t.value = t.value.strip('/')
        return t

    def t_INTEGER(self, t):
        r'/?\[\d+\]'
        t.value = int(re.search('\[(\d+)\]', t.value.strip('/')).group(1))
        return t

    def t_INTEGER_SET(self, t):
        r'/?\[(?:\d+,?)+\]'
        t.value = map(int, t.value.strip('[]').split(','))
        return t

    def t_INTERVAL(self, t):
        r'\[\d*\:\d*\]'
        t.value = self._parse_interval_str(re.search('\[(.+)\]', t.value).group(1))
        return t

    def t_STRING(self, t):
        r'/[^*/\[\]:\d][^+*/\[\]:]*'
        t.value = t.value.strip('/')
        return t

    def t_STRING_SET(self, t):
        r'/\[(?:[^+*/\[\]:\d][^+*/\[\]:]*,?)+\]'
        t.value = t.value.strip('/[]').split(',')
        return t

    def t_error(self, t):
        print 'Illegal character "%s"' % t.value[0]
        raise ValueError('Cannot tokenize selector')

    def p_list_plus(self, p):
        'list : list PLUS selector'
        p[0] = p[1]+[p[3]]

    def p_list_selector(self, p):
        'list : selector'
        p[0] = [p[1]]

    def p_selector_plus(self, p):
        'selector : selector token'
        p[0] = p[1]
        p[0].append(p[2])

    def p_selector_token(self, p):
        'selector : token'
        p[0] = [p[1]]

    def p_token(self, p):
        '''token : ASTERISK
               | INTEGER
               | INTEGER_SET
               | INTERVAL
               | STRING
               | STRING_SET'''
        p[0] = p[1]

    def p_error(self, p):
        print 'Syntax error'
        raise ValueError('Cannot parse selector')

    def _setup(self, **kwargs):
        """
        Build lexer and parser.
        """

        self.lexer = lex.lex(module=self, **kwargs)
        self.parser = yacc.yacc(module=self, **kwargs)

    def tokenize(self, selector):
        """
        Tokenize a selector string.

        Parameters
        ----------
        selector : str
            Row selector.

        Returns
        -------
        token_list : list
            List of tokens extracted by ply.
        """

        self.lexer.input(selector)
        token_list = []
        while True:
            token = self.lexer.token()
            if not token: break
            token_list.append(token)
        return token_list

    def parse(self, selector):
        """
        Parse a selector string.

        Parameters
        ----------
        selector : str
            Row selector.

        Returns
        -------
        parse_list : list of list
            List of lists containing the tokens corresponding to each individual
            selector in the string.
        """

        return self.parser.parse(selector, lexer=self.lexer)

    def max_levels(self, selector):
        """
        Return maximum number of token levels in selector.

        Parameters
        ----------
        selector : str
            Row selector.

        Returns
        -------
        count : int
            Maximum number of tokens in selector.
        """

        return max(map(len, self.parse(selector)))

    def _select_test(self, row, parse_list, start=None, stop=None):
        """
        Check whether the entries in a subinterval of a given tuple of data
        corresponding to the entries of one row in a DataFrame match the
        specified token values.

        Parameters
        ----------
        row : list
            List of data corresponding to a single row of a DataFrame.
        parse_list : list
            List of lists of token values extracted by ply.
        start, stop : int
            Start and end indices in `row` over which to test entries.

        Returns
        -------
        result : bool
            True of all entries in specified subinterval of row match, False otherwise.
        """

        row_sub = row[start:stop]
        for token_list in parse_list:

            # If this loop terminates prematurely, it will not return True; this 
            # forces checking of the subsequent token list:
            for i, token in enumerate(token_list):
                if token == '*':
                    continue
                elif type(token) in [int, str]:
                    if row_sub[i] != token:
                        break
                elif type(token) == list:
                    if row_sub[i] not in token:
                        break
                elif type(token) == tuple:
                    i_start, i_stop = token
                    if not(row_sub[i] >= i_start and row_sub[i] < i_stop):
                        break
                else:
                    continue
            else:
                return True

        # If the function still hasn't returned, no match was found:
        return False

    def get_tuples(self, df, selector, start=None, stop=None):
        """
        Return tuples containing MultiIndex labels selected by specified selector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : str
            Row selector.
        start, stop : int
            Start and end indices in `row` over which to test entries.

        Returns
        -------
        result : list
            List of tuples containing MultiIndex labels for selected rows.
        """

        parse_list = self.parse(selector)
        max_levels = max(map(len, parse_list))

        # The maximum number of tokens must not exceed the number of levels in the
        # DataFrame's MultiIndex:        
        if max_levels > len(df.index.names[start:stop]):
            raise ValueError('Maximum number of levels in selector exceeds that of '
                             'DataFrame index')

        return [t for t in df.index if self._select_test(t, parse_list,
                                                         start, stop)]

    def get_index(self, df, selector, start=None, stop=None, names=[]):
        """
        Return MultiIndex corresponding to rows selected by specified selector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : str
            Row selector.
        start, stop : int
            Start and end indices in `row` over which to test entries.
        names : list
            Names of levels to use in generated MultiIndex.

        Returns
        -------
        result : pandas.MultiIndex
            MultiIndex that refers to the rows selected by the specified
            selector.
        """

        tuples = self.get_tuples(df, selector, start, stop)
        if not tuples:
            raise ValueError('no tuples matching selector found')

        # XXX This probably could be made faster by directly manipulating the
        # existing MultiIndex:
        if names:
            return pd.MultiIndex.from_tuples(tuples, names=names)
        else:
            return pd.MultiIndex.from_tuples(tuples)

    def make_index(self, selector, names=[]):
        """
        Create a MultiIndex from the specified selector.

        Parameters
        ----------
        selector : str
            Row selector.
        names : list
            Names of levels to use in generated MultiIndex.

        Returns
        -------
        result : pandas.MultiIndex
            MultiIndex corresponding to the specified selector.

        Notes
        -----
        The selector may not contain any '*' character. It also must contain
        more than one level.
        """

        assert not re.search(r'\*', selector)
        parse_list = self.parse(selector)

        idx_list = []
        for token_list in parse_list:
            list_list = []
            for token in token_list:
                if type(token) == tuple:
                    list_list.append(range(token[0], token[1]))
                elif type(token) == list:
                    list_list.append(token)
                else:
                    list_list.append([token])
            if names:
                idx = pd.MultiIndex.from_product(list_list, names=names)
            else:
                idx = pd.MultiIndex.from_product(list_list)

            # XXX Attempting to run MultiIndex.from_product with an argument
            # containing a single list results in an Index, not a MultiIndex:
            assert type(idx) == pd.MultiIndex
            idx_list.append(idx)

        return reduce(type(idx_list[0]).append, idx_list)

    def select(self, df, selector, start=None, stop=None):
        """
        Select rows from DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : str
            Row selector.
        start, stop : int
            Start and end indices in `row` over which to test entries.

        Returns
        -------
        result : pandas.DataFrame
            DataFrame containing selected rows.
        """

        parse_list = self.parse(selector)

        # The number of tokens must not exceed the number of levels in the
        # DataFrame's MultiIndex:        
        if len(parse_list) > len(df.index.names[start:stop]):
            raise ValueError('Number of levels in selector exceeds number in row subinterval')

        return df.select(lambda row: self._select_test(row, parse_list, start, stop))

class PortMapper(object):
    """
    Maps a numpy array to/from path-like port identifiers.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data to map to ports.
    selectors : str or list of str
        Path-like selector(s) to map to `data`. If more than one selector is
        defined, the indices corresponding to each selector are sequentially 
        concatenated.
    """

    def __init__(self, data, selectors):

        # Can currently only handle unidimensional data structures:
        assert np.ndim(data) == 1
        assert type(data) == np.ndarray

        self.data = data
        self.sel = PathLikeSelector()
        self.portmap = pd.Series(data=np.arange(len(data)))
        if np.iterable(selectors) and type(selectors) is not str:
            idx_list = [self.sel.make_index(s) for s in selectors]
            idx = reduce(pd.MultiIndex.append, idx_list)
        else:
            idx = self.sel.make_index(selectors)
        self.portmap.index = idx

    def get(self, selector):
        """
        Retrieve mapped data specified by given selector.

        Parameters
        ----------
        selector : str
            Path-like selector.

        Returns
        -------
        result : numpy.ndarray
            Selected data.
        """
        
        return self.data[self.sel.select(self.portmap, selector).values]

    def set(self, selector, data):
        """
        Set mapped data specified by given selector.

        Parameters
        ----------
        selector : str
            Path-like selector.
        data : numpy.ndarray
            Array of data to save.
        """
        
        self.data[self.sel.select(self.portmap, selector).values] = data

    def __getitem__(self, selector):
        return self.get(selector)

    def __setitem__(self, selector, data):
        self.set(selector, data)

    def __repr__(self):
        return 'map:\n'+self.portmap.__repr__()+'\n\ndata:\n'+self.data.__repr__()

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

    class test_path_like_selector(TestCase):
        def setUp(self):
            self.df = pd.DataFrame(data={'data': np.random.rand(10),
                                         0: ['foo', 'foo', 'foo', 'foo', 'foo',
                                             'bar', 'bar', 'bar', 'baz', 'baz'],
                                         1: ['qux', 'qux', 'mof', 'mof', 'mof',
                                             'qux', 'qux', 'qux', 'qux', 'mof'],
                                         2: [0, 1, 0, 1, 2, 0, 1, 2, 0, 0]})
            self.df.set_index(0, append=False, inplace=True)
            self.df.set_index(1, append=True, inplace=True)
            self.df.set_index(2, append=True, inplace=True)
            self.sel = PathLikeSelector()
        def test_str_one(self):
            result = self.sel.select(self.df, '/foo')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_plus(self):
            result = self.sel.select(self.df, '/foo/qux+/baz/mof')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('baz','mof',0)], names=[0, 1, 2])
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

        def test_str_integer_set(self):
            result = self.sel.select(self.df, '/foo/qux[0,1]')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_string_set(self):
            result = self.sel.select(self.df, '/foo/[qux,mof]')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)],
                                            names=[0, 1, 2])
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

