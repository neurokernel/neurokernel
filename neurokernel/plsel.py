#!/usr/bin/env python

"""
Path-like row selector for pandas DataFrames with hierarchical MultiIndexes.
"""

import itertools
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
    corresponds to a level of the selector. An index level may either be a denoted by a
    string label (e.g., 'foo') or a numerical index (e.g., 0, 1, 2); a selector
    level may additionally be a list of strings (e.g., '[foo,bar]') or integers
    (e.g., '[0,2,4]') or continuous intervals (e.g., '[0:5]'). The '*' symbol
    matches any value in a level, while a range with an open upper bound (e.g.,
    '[5:]') will match all integers greater than or equal to the lower bound.
    
    Examples of valid selectors include

    /foo/bar
    /foo+/bar          (equivalent to /foo/bar)
    /foo/[qux,bar]
    /foo/bar[0]
    /foo/bar/[0]       (equivalent to /foo/bar[0])
    /foo/bar/0         (equivalent to /foo/bar[0])
    /foo/bar[0,1]
    /foo/bar[0:5]
    /foo/*/baz
    /foo/*/baz[5]
    /foo/bar,/baz/qux
    (/foo,/bar)+/baz   (equivalent to /foo/baz,/bar/baz)
    /[foo,bar].+/[0:2] (equivalent to /foo[0],/bar[1])

    The class can also be used to create new MultiIndex instances from selectors
    that can be fully expanded into an explicit set of identifiers (and
    therefore contain no ambiguous symbols such as '*' or '[:]').

    Methods
    -------
    aredisjoint(s0, s1, ...)
        Check whether several selectors are disjoint.
    expand(selector)
        Expand a nonambiguous selector into a list of identifiers.
    get_index(df, selector, start=None, stop=None, names=[])
        Return MultiIndex corresponding to rows selected by specified selector.
    get_tuples(df, selector, start=None, stop=None)
        Return tuples containing MultiIndex labels selected by specified selector.
    isambiguous(selector)
        Check whether a selector cannot be expanded into an explicit list of identifiers.
    make_index(selector, names=[])
        Create a MultiIndex from the specified selector.
    max_levels(selector)
        Return maximum number of token levels in selector.
    parse(selector)
        Parse a selector string into individual port identifiers.
    select(df, selector, start=None, stop=None)
        Select rows from DataFrame using a path-like selector.
    tokenize(selector)
        Tokenize a selector string.

    Notes
    -----
    Numerical indices are assumed to be zero-based. Intervals do not include the
    end element (i.e., like numpy, not like pandas).
    """

    tokens = ('ASTERISK', 'COMMA', 'DOTPLUS', 'INTEGER', 'INTEGER_SET',
              'INTERVAL', 'LPAREN', 'PLUS', 'RPAREN', 'STRING', 'STRING_SET')

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

    def t_DOTPLUS(self, t):
        r'\.\+'
        return t

    def t_COMMA(self, t):
        r'\,'
        return t

    def t_LPAREN(self, t):
        r'\('
        return t

    def t_RPAREN(self, t):
        r'\)'
        return t

    def t_ASTERISK(self, t):
        r'/\*'
        t.value = t.value.strip('/')
        return t

    def t_INTEGER(self, t):
        r'/?\d+'
        t.value = int(t.value.strip('/'))
        return t

    def t_INTEGER_SET(self, t):
        r'/?\[(?:\d+,?)+\]'
        t.value = map(int, t.value.strip('/[]').split(','))
        return t

    def t_INTERVAL(self, t):
        r'/?\[\d*\:\d*\]'
        t.value = self._parse_interval_str(re.search('\[(.+)\]', t.value).group(1))
        return t

    def t_STRING(self, t):
        r'/[^*/\[\]\(\):,\.\d][^+*/\[\]\(\):,\.]*'
        t.value = t.value.strip('/')
        return t

    def t_STRING_SET(self, t):
        r'/?\[(?:[^+*/\[\]\(\):,\.\d][^+*/\[\]\(\):,\.]*,?)+\]'
        t.value = t.value.strip('/[]').split(',')
        return t

    def t_error(self, t):
        raise ValueError('Cannot tokenize selector - illegal character: %s' % t.value[0])

    # A selector is a list of lists of levels:
    def p_selector_paren_selector(self, p):
        'selector : LPAREN selector RPAREN'
        p[0] = p[2]

    def p_selector_comma_selector(self, p):
        'selector : selector COMMA selector'
        p[0] = p[1]+p[3]

    def p_selector_plus_selector(self, p):
        'selector : selector PLUS selector'
        p[0] = [a+b for a, b in itertools.product(p[1], p[3])]

    def p_selector_dotplus_selector(self, p):
        'selector : selector DOTPLUS selector'
        # Expand ranges and wrap strings with lists in each selector:
        for i in xrange(len(p[1])): 
            for j in xrange(len(p[1][i])): 
                if type(p[1][i][j]) in [int, str]:
                    p[1][i][j] = [p[1][i][j]]
                elif type(p[1][i][j]) == tuple:
                    p[1][i][j] = range(p[1][i][j][0], p[1][i][j][1])
        for i in xrange(len(p[3])):
            for j in xrange(len(p[3][i])):
                if type(p[3][i][j]) in [int, str]:
                    p[3][i][j] = [p[3][i][j]]
                if type(p[3][i][j]) == tuple:
                    p[3][i][j] = range(p[3][i][j][0], p[3][i][j][1])
                    
        # Fully expand both selectors into individual identifiers
        ids_1 = [list(x) for y in p[1] for x in itertools.product(*y)]
        ids_3 = [list(x) for y in p[3] for x in itertools.product(*y)]
        
        # The expanded selectors must comprise the same number of identifiers:
        assert len(ids_1) == len(ids_3)        
        p[0] = [a+b for (a, b) in zip(ids_1, ids_3)]

    def p_selector_selector_plus_level(self, p):
        'selector : selector PLUS level'
        p[0] = [x+[p[3]] for x in p[1]]

    def p_selector_selector_level(self, p):
        'selector : selector level'
        p[0] = [x+[p[2]] for x in p[1]]

    def p_selector_level(self, p):
        'selector : level'
        p[0] = [[p[1]]]

    def p_level(self, p):
        '''level : ASTERISK
                 | INTEGER
                 | INTEGER_SET
                 | INTERVAL
                 | STRING
                 | STRING_SET'''
        p[0] = p[1]

    def p_error(self, p):
        raise ValueError('Cannot parse selector - syntax error: %s' % p)
        
    def _setup(self):
        """
        Build lexer and parser.
        """

        # Set the option optimize=1 in the production version:
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self, debug=0, write_tables=0)

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
        Parse a selector string into individual port identifiers.

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

    def isambiguous(self, selector):
        """
        Check whether a selector cannot be expanded into an explicit list of identifiers.

        A selector is ambiguous if it contains the symbols '*' or '[:]' (i.e., a
        range with no upper bound).

        Parameters
        ----------
        selector : str
            Selector to test.

        Returns
        -------
        result : bool
            True if the selector is ambiguous, False otherwise.
        """

        if re.search(r'(?:\*)|(?:\:\])', selector):
            return True
        else:
            return False

    def expand(self, selector):
        """
        Expand a nonambiguous selector into a list of identifiers.

        Parameters
        ----------
        selector : str
            Selector string.

        Returns
        -------
        result : list
            List of identifiers; each identifier is a list of tokens.
        """
        
        assert not self.isambiguous(selector)
        p = self.parse(selector)
        for i in xrange(len(p)):
            for j in xrange(len(p[i])):
                if type(p[i][j]) in [int, str]:
                    p[i][j] = [p[i][j]]
                elif type(p[i][j]) == tuple:
                    p[i][j] = range(p[i][j][0], p[i][j][1])
        return [list(x) for y in p for x in itertools.product(*y)]

    def isexpandable(self, selector):
        """
        Check whether a selector can be expanded into multiple identifiers.

        Parameters
        ----------
        selector : str
            Selector string.

        Returns
        -------
        result : bool
            True if the string contains any intervals or sets of
            strings/integers, False otherwise. Ambiguous selectors are
            not deemed to be expandable.
        """

        if self.isambiguous(selector):
            return False
        p = self.parse(selector)

        for i in xrange(len(p)):
            for j in xrange(len(p[i])):
                if type(p[i][j]) in [int, str]:
                    p[i][j] = [p[i][j]]
                elif type(p[i][j]) == tuple:
                    p[i][j] = range(p[i][j][0], p[i][j][1])

                    # The presence of a range containing more than 1 element
                    # ensures expandability:
                    if len(p[i][j]) > 1:
                        return True
                elif type(p[i][j]) == list:

                    # The presence of a list containing more than 1 unique
                    # element ensures expandability:
                    if len(set(p[i][j])) > 1: return True

        if len(set([tuple(x) for y in p for x in itertools.product(*y)])) > 1:
            return True
        else:
            return False

    def collapse(self, id_list):
        """
        Collapse a list of identifiers into a selector string.

        Parameters
        ----------
        id_list : list
            List of identifiers; each identifier is a list of tokens.

        Returns
        -------
        selector : str
            String that expands into the given identifier list.

        Notes
        -----
        Expects all identifiers in the given list to have the same
        number of levels.
        """

        # Can only collapse list identifiers that all have the same number of
        # levels:
        assert len(set(map(len, id_list))) == 1

        # Collect all tokens for each level:

        # If a level contains multiple string AND integer tokens, convert it to
        # a list:

        # If a level contains nonconsecutive integers, convert it into a list:

        # If a level only contains consecutive integers, convert it into an
        # interval:

        # XXX unfinished

    def aredisjoint(self, *selectors):
        """
        Check whether several selectors are disjoint.

        Parameters
        ----------
        s0, s1, ... : str
            Selectors to check.

        Returns
        -------
        result : bool
            True if none of the identifiers comprised by one selector are
            comprised by the other.

        Notes
        -----
        The selectors must not be ambiguous.
        """

        assert len(selectors) >= 1
        if len(selectors) == 1:
            return True
        assert all(map(lambda s: not self.isambiguous(s), selectors))

        # Expand selectors into sets of identifiers:
        ids = set()
        for selector in selectors:

            # If some identifiers are present in both the previous expanded
            # selectors and the current selector, the selectors cannot be disjoint:
            ids_new = set(map(tuple, self.expand(selector)))
            if ids.intersection(ids_new):
                return False
            else:
                ids = ids.union(ids_new)
        return True

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

        try:
            return self.max_levels.cache[selector]
        except:
            count = max(map(len, self.parse(selector)))
            self.max_levels.cache[selector] = count
            return count
    max_levels.cache = {}

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
        The selector may not contain ambiguous symbols such as '*' or 
        '[:]'. It also must contain more than one level.        
        """

        assert not self.isambiguous(selector)
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

            # Attempting to run MultiIndex.from_product with an argument
            # containing a single list results in an Index, not a MultiIndex:
            assert type(idx) == pd.MultiIndex

            idx_list.append(idx)

        # All of the token lists in the selector must have the same number of
        # levels:
        assert len(set(map(lambda idx: len(idx.levels), idx_list))) == 1

        return reduce(pd.MultiIndex.append, idx_list)

    def select(self, df, selector, start=None, stop=None):
        """
        Select rows from DataFrame using a path-like selector.

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
    selector : str
        Selector to map to `data`.
    idx : sequence
        Indices of elements in the specified array to map to ports. If no
        indices are specified, the entire array is mapped to the ports specified
        by the given selector.

    Notes
    -----
    The selectors may not contain any '*' or '[:]' characters.
    """

    def __init__(self, data, selectors, idx=None):

        # Can currently only handle unidimensional data structures:
        assert np.ndim(data) == 1
        assert type(data) == np.ndarray

        # Save a reference to the specified array:
        self.data = data

        self.sel = PathLikeSelector()
        if idx is None:
            self.portmap = pd.Series(data=np.arange(len(data)))
        else:
            self.portmap = pd.Series(data=np.asarray(idx))
        idx = self.sel.make_index(selector)
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

    __getitem__ = get
    __setitem__ = set

    def __repr__(self):
        return 'map:\n'+self.portmap.__repr__()+'\n\ndata:\n'+self.data.__repr__()

if __name__ == '__main__':
    from unittest import main, TestCase
    from pandas.util.testing import assert_frame_equal

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

    class test_path_like_selector(TestCase):
        def setUp(self):
            self.df = df.copy()
            self.sel = PathLikeSelector()

        def test_str(self):
            result = self.sel.select(self.df, '/foo')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_comma(self):
            result = self.sel.select(self.df, '/foo/qux,/baz/mof')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('baz','mof',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_plus(self):
            result = self.sel.select(self.df, '/foo+/qux+[0,1]')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_dotplus(self):
            result = self.sel.select(self.df, '/[bar,baz].+/[qux,mof].+/[0,0]')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('baz','mof',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_paren(self):
            result = self.sel.select(self.df, '(/bar,/baz)')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('bar','qux',1),
                                             ('bar','qux',2),
                                             ('baz','qux',0),
                                             ('baz','mof',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_paren_plus(self):
            result = self.sel.select(self.df, '(/bar,/baz)+/qux')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('bar','qux',1),
                                             ('bar','qux',2),
                                             ('baz','qux',0)], names=[0, 1, 2])
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

        def test_str_integer_with_brackets(self):
            result = self.sel.select(self.df, '/bar/qux[1]')
            idx = pd.MultiIndex.from_tuples([('bar','qux',1)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_integer_no_brackets(self):
            result = self.sel.select(self.df, '/bar/qux/1')
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

        def test_str_interval_no_bounds(self):
            result = self.sel.select(self.df, '/foo/mof[:]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_interval_lower_bound(self):
            result = self.sel.select(self.df, '/foo/mof[1:]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',1),
                                             ('foo','mof',2)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_interval_upper_bound(self):
            result = self.sel.select(self.df, '/foo/mof[:2]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',0),
                                             ('foo','mof',1)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_str_interval_both_bounds(self):
            result = self.sel.select(self.df, '/bar/qux[0:2]')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('bar','qux',1)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

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

    main()

