#!/usr/bin/env python

"""
Path-like row selector for pandas DataFrames with hierarchical MultiIndexes.
"""

import pdb
import copy
import itertools
import re
import sys

from past.builtins import basestring, long
import numpy as np
import pandas as pd
import ply.lex as lex
import ply.yacc as yacc

# Work around lack of support for serializing slices in msgpack 0.4.4:
def _encode(obj):
    if isinstance(obj, slice):
        return {'type': 'slice',
                'data': (obj.start, obj.stop, obj.step)}
    else:
        return obj

def _decode(obj):
    try:
        if obj['type'] == 'slice':
            return slice(*obj['data'])
        else:
            return obj
    except:
        return obj

# Fall back to using pickle for hashing if msgpack isn't available:
try:
    import msgpack
except ImportError:
    if sys.version_info.major == 2:
        import cPickle as pickle
    else:
        import pickle
    _packb = pickle.dumps
    _unpackb = pickle.loads
else:
    _packb = lambda x: msgpack.packb(x, default=_encode)
    _unpackb = lambda x: msgpack.unpackb(x, object_hook=_decode)

class Selector(object):
    """
    Validated and expanded port selector.

    Parameters
    ----------
    s : Selector, str, or unicode
        Existing Selector class instance or string representation.
        The selector may not be ambiguous. If an existing Selector instance
        is specified, the new instance is a copy of the existing instance.

    Attributes
    ----------
    str : str
        String representation of selector.
    expanded : tuple of tuples
        Expanded selector.
    max_levels : int
        Maximum number of levels in selector.
    """

    def __init__(self, s):
        if isinstance(s, Selector):
            self._expanded = copy.copy(s._expanded)
            self._max_levels = copy.copy(s._max_levels)
        elif isinstance(s, basestring):

            # Save expanded selector as tuple because it shouldn't need to be
            # modified after expansion:
            self._expanded = tuple(SelectorMethods.expand(s))
            self._max_levels = max(map(len, self._expanded))
        else:
            self._expanded = tuple(SelectorMethods.expand(s))
            self._max_levels = max(map(len, self._expanded))

    @property
    def nonempty(self):
        """
        True if the selector contains identifiers.
        """

        return bool(self._max_levels)

    @property
    def str(self):
        """
        String representation of selector.
        """

        return SelectorMethods.collapse(self._expanded)

    @property
    def expanded(self):
        """
        Expanded selector.
        """

        return self._expanded

    @property
    def identifiers(self):
        """
        List of individual identifiers in selector.
        """

        return [SelectorMethods.collapse((i,)) for i in self._expanded]

    @property
    def max_levels(self):
        """
        Maximum number of levels in selector.
        """

        return self._max_levels

    @classmethod
    def add(cls, *sels):
        """
        Combine the identifiers in multiple selectors into a single selector.

        Parameters
        ----------
        sels : iterable of Selector
            Selector instances.

        Returns
        -------
        result : Selector
            Selector containing all of the port identifiers comprised by all of the
            arguments.

        Notes
        -----
        Duplicate identifiers are not omitted.
        """

        out = cls('')
        try:
            out._max_levels = max([s.max_levels for s in sels if s.nonempty])
        except ValueError:
            out._max_levels = 0

        out._expanded = tuple(i for s in sels \
                for i in s._expanded if s.nonempty) or ((),)
        return out

    @classmethod
    def add_str(cls, *s):
        """
        Combine the identifiers in multiple selector strings into a single selector.
        """

        return cls.add(*map(cls, s))

    @classmethod
    def concat(cls, *sels):
        """
        Concatenate the identifiers in multiple selectors elementwise.

        Parameters
        ----------
        sels : iterable of Selector
            Selector instances.

        Returns
        -------
        result : Selector
            Each port identifier in the returned Selector is equivalent to
            the elementwise concatenation of the identifiers in the listed
            Selector instances.
        """

        out = cls('')
        s_len = None
        e_list = []
        for s in sels:
            if s_len is None:
                s_len = len(s)
            else:
                assert len(s) == s_len
            if not e_list:
                e_list = list(list(t) for t in s._expanded)
            else:
                for e, t in zip(e_list, s._expanded):
                    e.extend(t)
        out._expanded = tuple(tuple(e) for e in e_list)
        out._max_levels = sum([s.max_levels for s in sels if s.nonempty])
        return out

    @classmethod
    def prod(cls, *sels):
        """
        Compute the product of identifiers in multiple selectors.

        Parameters
        ----------
        sels : iterable of Selector
            Selector instances.

        Returns
        -------
        result : Selector
            Selector containing all of the port identifiers comprised by the
            product of all of the arguments.

        Notes
        -----
        Duplicate identifiers are not omitted.
        """

        out = cls('')
        out._max_levels = sum([s.max_levels for s in sels if s.nonempty])
        out._expanded = tuple(tuple(j for j in itertools.chain(*i)) \
                for i in itertools.product(*[s.expanded for s in sels]))
        return out

    @classmethod
    def union(cls, *sels):
        """
        Compute the union of the identifiers in multiple selectors.

        Parameters
        ----------
        sels : iterable of Selector
            Selector instances.

        Returns
        -------
        result : Selector
            Selector containing all of the port identifiers comprised by the
            union of all of the arguments.
        """

        out = cls('')
        tmp = set()
        for s in sels:
            if s.nonempty:
                tmp = tmp.union(s.expanded)
        if tmp:
            out._expanded = tuple(sorted(tmp))
        else:
            out._expanded = ((),)
        try:
            out._max_levels = max([s.max_levels for s in sels if s.nonempty])
        except ValueError:
            out._max_levels = 0
        return out

    def __add__(self, y):
        return self.add(self, y)

    def __len__(self):
        if len(self._expanded) == 1 and not self._expanded[0]:
            return 0
        else:
            return len(self._expanded)

    def __iter__(self):
        if self.nonempty:
            for t in self._expanded:
                yield (t,)
        else:
            yield ((),)

    def __repr__(self):
        s = self.str
        if len(s) <= 100:
            return 'Selector(\'%s\')' % s
        else:
            return 'Selector(\'%s\')' % (s[0:25]+' ... '+s[-25:])

class SelectorParser(object):
    """
    This class implements a parser for path-like selectors that can
    be associated with elements in a sequential data structure such as a
    Pandas DataFrame; in the latter case, each level of the selector corresponds
    to a level of a Pandas MultiIndex. An index level may either be a
    denoted by a string label (e.g., 'foo') or a numerical index (e.g., 0, 1,
    2); a selector level may additionally be a list of strings (e.g.,
    '[foo,bar]') or integers (e.g., '[0,2,4]') or continuous intervals
    (e.g., '[0:5]'). The '*' symbol matches any value in a level, while a
    range with an open upper bound (e.g., '[5:]') will match all integers
    greater than or equal to the lower bound.

    Examples of valid selectors include

    ==================  =================================
    Selector            Comments
    ==================  =================================
    /foo/bar
    /foo+/bar           equivalent to /foo/bar
    /foo/[qux,bar]
    /foo/bar[0]
    /foo/bar/[0]        equivalent to /foo/bar[0]
    /foo/bar/0          equivalent to /foo/bar[0]
    /foo/bar[0,1]
    /foo/bar[0:5]
    /foo/*/baz
    /foo/*/baz[5]
    /foo/bar,/baz/qux
    (/foo,/bar)+/baz    equivalent to /foo/baz,/bar/baz
    /[foo,bar].+/[0:2]  equivalent to /foo[0],/bar[1]
    ==================  =================================

    Notes
    -----
    An empty string is deemed to be a valid selector.

    Since there is no need to maintain multiple instances of the lexer/parser
    used to process path-like selectors, they are associated with the class
    rather than class instances; likewise, all of the class' methods are
    classmethods.

    Numerical indices in selectors are assumed to be
    zero-based. Intervals do not include the end element (i.e., like numpy, not
    like Pandas).
    """

    tokens = ('ASTERISK', 'COMMA', 'DOTPLUS', 'INTEGER', 'INTEGER_SET',
              'INTERVAL', 'LPAREN', 'PLUS', 'RPAREN', 'STRING', 'STRING_SET')

    @classmethod
    def _parse_interval_str(cls, s):
        """
        Convert string representation of interval to slice.
        """

        start, stop = s.split(':')
        if start == '':
            start = 0
        else:
            start = int(start)
        if stop == '':
            stop = None
        else:
            stop = int(stop)
        return slice(start, stop)

    @classmethod
    def t_PLUS(cls, t):
        r'\+'
        return t

    @classmethod
    def t_DOTPLUS(cls, t):
        r'\.\+'
        return t

    @classmethod
    def t_COMMA(cls, t):
        r'\,'
        return t

    @classmethod
    def t_LPAREN(cls, t):
        r'\('
        return t

    @classmethod
    def t_RPAREN(cls, t):
        r'\)'
        return t

    @classmethod
    def t_ASTERISK(cls, t):
        r'/\*'
        t.value = t.value.strip('/')
        return t

    @classmethod
    def t_INTEGER(cls, t):
        r'/?\d+'
        t.value = int(t.value.strip('/'))
        return t

    @classmethod
    def t_INTEGER_SET(cls, t):
        r'/?\[(?:\d+,?)+\]'
        t.value = list(map(int, t.value.strip('/[]').split(',')))
        return t

    @classmethod
    def t_INTERVAL(cls, t):
        r'/?\[\d*\:\d*\]'
        t.value = cls._parse_interval_str(re.search('\[(.+)\]', t.value).group(1))
        return t

    @classmethod
    def t_STRING(cls, t):
        r'/[^*/\[\]\(\):,\.\d][^+*/\[\]\(\):,\.]*'
        t.value = t.value.strip('/')
        return t

    @classmethod
    def t_STRING_SET(cls, t):
        r'/?\[(?:[^+*/\[\]\(\):,\.\d][^+*/\[\]\(\):,\.]*,?)+\]'
        t.value = t.value.strip('/[]').split(',')
        return t

    @classmethod
    def t_error(cls, t):
        raise ValueError('Cannot tokenize selector - illegal character: %s' % t.value[0])

    # A selector is a list of lists of levels:
    @classmethod
    def p_selector_paren_selector(cls, p):
        'selector : LPAREN selector RPAREN'
        p[0] = p[2]

    @classmethod
    def p_selector_comma_selector(cls, p):
        'selector : selector COMMA selector'
        p[0] = p[1]+p[3]

    @classmethod
    def p_selector_plus_selector(cls, p):
        'selector : selector PLUS selector'
        p[0] = [a+b for a, b in itertools.product(p[1], p[3])]

    @classmethod
    def p_selector_dotplus_selector(cls, p):
        'selector : selector DOTPLUS selector'
        # Expand ranges and wrap strings with lists in each selector:
        for i in range(len(p[1])):
            for j in range(len(p[1][i])):
                if type(p[1][i][j]) in [int, str]:
                    p[1][i][j] = [p[1][i][j]]
                elif type(p[1][i][j]) == slice:
                    p[1][i][j] = range(p[1][i][j].start, p[1][i][j].stop)
        for i in range(len(p[3])):
            for j in range(len(p[3][i])):
                if type(p[3][i][j]) in [int, str]:
                    p[3][i][j] = [p[3][i][j]]
                elif type(p[3][i][j]) == slice:
                    p[3][i][j] = range(p[3][i][j].start, p[3][i][j].stop)

        # Fully expand both selectors into individual identifiers
        ids_1 = [list(x) for y in p[1] for x in itertools.product(*y)]
        ids_3 = [list(x) for y in p[3] for x in itertools.product(*y)]

        # The expanded selectors must comprise the same number of identifiers:
        assert len(ids_1) == len(ids_3)
        p[0] = [a+b for (a, b) in zip(ids_1, ids_3)]

    @classmethod
    def p_selector_selector_plus_level(cls, p):
        'selector : selector PLUS level'
        p[0] = [x+[p[3]] for x in p[1]]

    @classmethod
    def p_selector_selector_level(cls, p):
        'selector : selector level'
        p[0] = [x+[p[2]] for x in p[1]]

    @classmethod
    def p_selector_level(cls, p):
        'selector : level'
        p[0] = [[p[1]]]

    @classmethod
    def p_level(cls, p):
        '''level : ASTERISK
                 | INTEGER
                 | INTEGER_SET
                 | INTERVAL
                 | STRING
                 | STRING_SET'''
        p[0] = p[1]

    @classmethod
    def p_error(cls, p):
        raise ValueError('Cannot parse selector - syntax error: %s' % p)

    @classmethod
    def tokenize(cls, selector):
        """
        Tokenize a selector string.

        Parameters
        ----------
        selector : str
            Selector string.

        Returns
        -------
        token_list : list
            List of tokens extracted by ply.
        """

        cls.lexer.input(selector)
        token_list = []
        while True:
            token = cls.lexer.token()
            if not token: break
            token_list.append(token)
        return token_list

    @classmethod
    def pad_parsed(cls, selector, pad_len=float('inf'), inplace=True):
        """
        Pad token lists in a parsed selector to some maximum length.

        Parameters
        ----------
        selector : list of lists
            Parsed selector.
        pad_len : int
            Final length of each token list. If set to Inf, all tokens are
            padded to the maximum token length.
        inplace : bool
            If True, modify the selector in place; otherwise, return a modified
            copy.

        Returns
        -------
        result : list of lists
            Padded selector.
        """

        assert isinstance(selector, list)
        if pad_len == float('inf'):
            pad_len = max(map(len, selector))
        if not inplace:
            selector = copy.deepcopy(selector)
        for x in selector:
            x += ['']*(pad_len-len(x))
        return selector

    @classmethod
    def parse(cls, selector, pad_len=0):
        """
        Parse a selector string into tokens.

        Parameters
        ----------
        selector : str
            Selector string.
        pad_len : int
            Length to which expanded token sequences should be padded with blanks.
            If infinite, the sequences are padded to the length of the longest
            sequence.

        Returns
        -------
        result : list of list
            List of lists containing the tokens corresponding to each individual
            selector in the string.

        Notes
        -----
        This method does not expand selectors into the tokens corresponding to
        individual port identifiers.

        See Also
        --------
        SelectorMethods.expand
        """

        if re.search('^\s*$', selector):
            result = [[]]
        else:
            result = cls.parser.parse(selector, lexer=cls.lexer)
        return cls.pad_parsed(result, pad_len)

class SelectorMethods(SelectorParser):
    """
    Class for manipulating and using path-like selectors.

    Contains class methods for expanding selectors, selecting rows from a
    Pandas DataFrame using a selector, etc.

    The class can also be used to create new MultiIndex instances from selectors
    that can be fully expanded into an explicit set of identifiers (and
    therefore contain no ambiguous symbols such as '*' or '[:]').
    """

    @classmethod
    def is_identifier(cls, s):
        """
        Check whether a selector or token sequence can identify a single port.

        Parameters
        ----------
        s : Selector, str, unicode, or sequence
            Selector class instance, raw selector string (e.g., '/foo[0:2]'),
            sequence of token sequences (e.g., [['foo', (0, 2)]]), or sequence
            of tokens (e.g., ['foo', 0]).

        Returns
        -------
        result : bool
            True for a sequence containing only strings and/or integers
            (e.g., ['foo', 0]) or a selector string that expands into a
            single sequence of strings and/or integers (e.g., [['foo', 0]]).

        Notes
        -----
        Can check sequences of tokens (even though a sequence of tokens is not a
        valid selector).
        """

        if isinstance(s, Selector):
            return len(s) == 1

        if np.iterable(s):

            # Try to expand string:
            if isinstance(s, basestring):
                try:
                    s_exp = cls.expand(s)
                except:
                    return False
                else:
                    if len(s_exp) == 1:
                        return True
                    else:
                        return False

            # If all entries are lists or tuples, try to expand:
            elif all([(isinstance(x, (list, slice))) for x in s]):
                if len(cls.expand(s)) == 1:
                    return True
                else:
                    return False

            # A sequence of integers and/or strings is a valid port identifier:
            elif all(map(lambda x: isinstance(x, (int, long, basestring)), s)):
                #elif set(map(type, s)).issubset([int, basestring]):
                return True
            else:
                return False

        # A non-iterable cannot be a valid identifier:
        else:
            return False

    @classmethod
    def to_identifier(cls, s):
        """
        Convert an expanded selector/token sequence into a single port identifier string.

        Parameters
        ----------
        s : sequence
            Expanded selector (i.e., a sequence of sequences) or a sequence of
            string or integer tokens.

        Returns
        -------
        s : str
            Port identifier string.

        Notes
        -----
        Accepts sequences of tokens as well as expanded selectors (even though
        a sequence of tokens is not a valid selector).
        """

        assert isinstance(s, (list, tuple))
        # if set(map(type, s)).issubset([int, long, str, basestring]):
        if all(map(lambda x: isinstance(x, (int, long, basestring)), s)):
            tokens = s
        else:
            assert len(s) == 1
            tokens = s[0]

        result = ''
        for t in tokens:
            if isinstance(t, basestring):
                result += '/'+t
            elif isinstance(t, (int, long)):
                result += '[%s]' % t
            else:
                raise ValueError('Cannot convert to single port identifier.')
        return result

    @classmethod
    def is_ambiguous(cls, selector):
        """
        Check whether a selector cannot be expanded into an explicit list of identifiers.

        A selector is ambiguous if it contains the symbols '*' or ':]' (i.e., a
        range with no upper bound).

        Parameters
        ----------
        selector : Selector, str, unicode or sequence
            Selector class instance, selector string (e.g., '/foo[0:2]'),
            or sequence of token sequences (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : bool
            True if the selector is ambiguous, False otherwise.
        """

        # The Selector class can only encapsulate an unambiguous selector:
        if isinstance(selector, Selector):
            return False

        if isinstance(selector, basestring):
            if re.search(r'(?:\*)|(?:\:\])', selector):
                return True
            else:
                return False
        elif type(selector) in [list, tuple]:
            for tokens in selector:
                for token in tokens:
                    if token == '*' or \
                       (type(token) == slice and token.stop is None):
                        return True
            return False
        else:
            raise ValueError('invalid selector type')

    @classmethod
    def is_selector_empty(cls, selector):
        """
        Check whether a string or sequence is an empty selector.

        Parameters
        ----------
        s : str, unicode, or sequence
            String or sequence to test.

        Returns
        -------
        result : bool
            True if `s` is a sequence containing empty sequences or a null
            string, False otherwise.

        Notes
        -----
        Ambiguous selectors are not deemed to be empty.
        """

        if isinstance(selector, Selector):
            return len(selector) == 0

        if isinstance(selector, basestring) and \
           re.search('^\s*$', selector):
            return True
        if type(selector) in [list, tuple] and \
             all([len(x) == 0 for x in selector]):
                return True
        return False

    @classmethod
    def is_selector_seq(cls, s):
        """
        Check whether a sequence is a valid selector.

        Parameters
        ----------
        s : sequence
            Sequence to test.

        Returns
        -------
        result : bool
            True if a sequence of valid token sequences
            (e.g., [['foo', (0, 2)]], [['bar', 'baz'], ['qux', 0]]),
            False otherwise.

        Note
        ----
        An empty sequence (e.g., []) is deemed to be a valid selector.
        """

        assert np.iterable(s)
        for tokens in s:

            # The selector must contain sequences of tokens:
            if not np.iterable(tokens):
                return False

            # Each token must either be a string, integer, slice,
            # list of strings, or list of integers:
            for token in tokens:
                if type(token) == list:
                    # token_types = set(map(type, token))
                    # if not (token_types.issubset([str, int])):
                    if not all(map(
                            lambda x: isinstance(x, (int, long, basestring)),
                            token)):
                        return False
                elif not isinstance(token, (slice, basestring, int, long)):
                    # elif not type(token) not in [slice, str, int]:
                    return False

        # All tokens are valid:
        return True

    @classmethod
    def is_selector_str(cls, s):
        """
        Check whether a string is a valid selector.

        Parameters
        ----------
        s : str, unicode
            String to test.

        Returns
        -------
        result : bool
            True if the specified selector is a parseable string
            (e.g., '/foo[0:2]'), False otherwise.
        """

        # assert type(s) is str
        assert isinstance(s, basestring)
        try:
            cls.parse(s)
        except:
            return False
        else:
            return True

    @classmethod
    def is_selector(cls, s):
        """
        Check whether a string or sequence is a valid selector.

        Parameters
        ----------
        s : Selector, str, unicode, or sequence
            Selector instance, string, or sequence to test.

        Returns
        -------
        result : bool
            True if the specified selector is a parseable string (e.g.,
            '/foo[0:2]') or a sequence of valid token sequences.
            (e.g., [['foo', (0, 2)]], [['bar', 'baz'], ['qux', 0]]).
        """

        if isinstance(s, Selector):
            return True
        # elif type(s) is str:
        elif isinstance(s, basestring):
            return cls.is_selector_str(s)
        elif np.iterable(s):
            return cls.is_selector_seq(s)
        else:
            return False

    @classmethod
    def expand(cls, selector, pad_len=0):
        """
        Expand an unambiguous selector into a list of identifiers.

        Parameters
        ----------
        selector : Selector, str, unicode, or sequence
            Selector class instance, string (e.g., '/foo[0:2]'), or sequence
            of token sequences (e.g., [['foo', (0, 2)]]).
        pad_len : int
            Length to which expanded token sequences should be padded with blanks.
            If infinite, the sequences are padded to the length of the longest
            sequence.

        Returns
        -------
        result : list
            List of identifiers. If the number of levels in the selector is 1,
            each is a string or integer token; otherwise, each identifier is
            a tuple of identifier is a tuple of tokens.

        Examples
        --------
        >>> from neurokernel.plsel import SelectorMethods
        >>> SelectorMethods.expand('/foo[0:2]')
        [('foo', 0), ('foo', 1)]
        >>> SelectorMethods.expand('/foo,/bar[0:2]', float('inf'))
        [('foo', ''), ('bar', 0), ('bar', 1)]
        >>> SelectorMethods.expand('/foo[0:2]', 3)
        [('foo', 0, ''), ('foo', 1, '')]
        >>> SelectorMethods.expand('/bar,/foo[0:2]', 3)
        [('bar', '', ''), ('foo', 0, ''), ('foo', 1, '')]
        """

        if isinstance(selector, Selector):
            if pad_len == 0:
                return selector.expanded
            elif pad_len == float('inf'):
                return [tuple(x)+('',)*(selector.max_levels-len(x)) \
                        for x in selector.expanded]
            else:
                return [tuple(x)+('',)*(pad_len-len(x)) \
                        for x in selector.expanded]

        #assert cls.is_selector(selector)
        assert not cls.is_ambiguous(selector)

        #if type(selector) is str:
        if isinstance(selector, basestring):
            try:
                p = cls.parse(selector)
            except:
                raise ValueError('invalid selector')
        elif np.iterable(selector):
            assert cls.is_selector(selector)
            # Assume empty iterables are empty selectors:
            if not selector:
                selector = [()]

            # Copy the selector to avoid modifying it:
            p = copy.copy(selector)
        else:
            raise ValueError('invalid selector type')

        max_levels = 0
        temp = []
        for i in range(len(p)):
            t = list(p[i])
            len_p = len(p[i])
            max_levels = max(max_levels, len_p)
            for j in range(len_p):

                # Wrap integers and strings in a list so that
                # itertools.product() can iterate over them:
                # if type(t[j]) in [int, str]:
                if isinstance(t[j], (int, long, basestring)):
                    t[j] = [t[j]]

                # Expand slices into ranges:
                elif type(t[j]) == slice:
                    t[j] = range(t[j].start, t[j].stop)
            temp.append(t)

        if pad_len == float('inf'):
            result = [tuple(x)+('',)*(max_levels-len(x)) \
                      for y in temp for x in itertools.product(*y)]
        else:
            result = [tuple(x)+('',)*(pad_len-len(x)) \
                      for y in temp for x in itertools.product(*y)]

        # If the selector doesn't expand to anything, return a list containing
        # an empty tuple:
        if result:
            return result
        else:
            return [()]

    @classmethod
    def is_expandable(cls, selector):
        """
        Check whether a selector can be expanded into multiple identifiers.

        Parameters
        ----------
        selector : Selector, str, unicode, or sequence
            Selector class instance, string (e.g., '/foo[0:2]'), or
            sequence of token sequences (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : bool
            True if the selector contains any intervals or sets of
            strings/integers, False otherwise. Ambiguous selectors are
            not deemed to be expandable, nor are fully expanded selectors or
            Selector instances.
        """

        assert cls.is_selector(selector)

        if isinstance(selector, Selector) or cls.is_ambiguous(selector):
            return False
        # if type(selector) is str:
        if isinstance(selector, basestring):
            p = cls.parse(selector)
        elif type(selector) in [list, tuple]:
            p = selector
        else:
            raise ValueError('invalid selector type')
        for i in range(len(p)):
            for j in range(len(p[i])):
                # if type(p[i][j]) in [int, str]:
                if isinstance(p[i][j], (int, long, basestring)):
                    p[i][j] = [p[i][j]]

                elif type(p[i][j]) == slice:
                    p[i][j] = range(p[i][j].start, p[i][j].stop)

                    # The presence of a range containing more than 1 element
                    # implies expandability:
                    if len(p[i][j]) > 1: return True
                elif type(p[i][j]) == list:

                    # The presence of a list containing more than 1 unique
                    # element implies expandability:
                    if len(set(p[i][j])) > 1: return True
                else:
                    raise ValueError('invalid selector contents')

        if len(set([tuple(x) for y in p for x in itertools.product(*y)])) > 1:
            return True
        else:
            return False

    @staticmethod
    def are_consecutive(int_list):
        """
        Check whether a list of integers is consecutive.

        Parameters
        ----------
        int_list : list of int
            List of integers

        Returns
        -------
        result : bool
            True if the integers are consecutive, false otherwise.

        Notes
        -----
        Does not assume that the list is sorted.
        """

        if set(np.diff(int_list)) == set([1]):
            return True
        else:
            return False

    @classmethod
    def tokens_to_str(cls, tokens):
        """
        Convert expanded/parsed token sequence into a single selector string.

        Parameters
        ----------
        s : sequence
            Sequence of expanded selector tokens.

        Returns
        -------
        result : str
            String corresponding to selector tokens.
        """

        assert np.iterable(tokens)
        result = []
        for t in tokens:
            # if type(t) in [str,  int]:
            if isinstance(t, (int, long, basestring)):
                result.append('/'+str(t))
            elif type(t) == slice:
                start = str(t.start) if t.start is not None else ''
                stop = str(t.stop) if t.stop is not None else ''
                result.append('[%s:%s]' % (start, stop))
            elif type(t) in [tuple, list]:
                if not t:
                    raise ValueError('invalid token')
                result.append('['+','.join(map(str, t))+']')
            else:
                raise ValueError('invalid token')
        return ''.join(result)

    @classmethod
    def collapse(cls, selector):
        """
        Collapse a selector into a single string.

        Parameters
        ----------
        selector : iterable
            Expanded selector. If the selector is a string, it is returned
            unchanged.

        Returns
        -------
        s : str
            String that comprises all identifiers in the specified expanded
            selector.
        """

        if isinstance(selector, basestring):
            return selector
        if isinstance(selector, Selector):
            return selector.str
        assert np.iterable(selector)
        result_list = []
        for tokens in selector:
            result_list.append(cls.tokens_to_str(tokens))
        return ','.join(result_list)

    @classmethod
    def _collapse(cls, id_list):
        """
        Collapse a list of identifiers into a selector string.

        Parameters
        ----------
        id_list : list of tuple
            List of identifiers; each identifier is a list of token tuples.

        Returns
        -------
        selector : str
            String that expands into the given identifier list.

        Notes
        -----
        Expects all identifiers in the given list to have the same
        number of levels.
        """

        # XXX doesn't collapse expanded selectors such as /foo/xxx,/bar/yyy
        # properly
        raise NotImplemented('unfinished method - should eventually replace collapse()')

        # Can only collapse list identifiers that all have the same number of
        # levels:
        assert len(set(map(len, id_list))) == 1

        # Collect all tokens for each level:
        levels = [[] for i in range(max(map(len, id_list)))]
        for i in range(len(id_list)):
            for j in range(len(id_list[i])):
                if not(id_list[i][j] in levels[j]):
                    levels[j].append(id_list[i][j])

        def collapse_level(level):
            """
            Recursively called function to collapse all values in a single level.
            """

            # type_set = set(map(type, level))
            # if type_set in set([int]):
            if all(map(lambda x: isinstance(x, (int, long)), level)):

                # If a level only contains consecutive integers, convert it into an
                # interval:
                level.sort()
                if cls.are_consecutive(level):
                    return ['[%s:%s]' % (min(level), max(level)+1)]

                # If a level contains nonconsecutive integers, convert it into a
                # list:
                else:
                    return ['['+','.join([str(i) for i in level])+']']

            elif all(map(lambda x: isinstance(x, basestring), level)):
                # elif type_set in set([str]):
                if len(level) == 1:
                    return level
                else:
                    return ['['+','.join([s for s in level])+']']
            else:
                level_int = sorted([x for x in level if isinstance(x, (int, long))])
                level_str = sorted([x for x in level if isinstance(x, basestring)])
                return collapse_level(level_int)+collapse_level(level_str)

        # If a level contains multiple string AND integer tokens, convert it to
        # a list:
        collapsed_list = []
        for level in levels:
            collapsed_list.append(collapse_level(sorted(level)))
        selector_list = []
        for t in itertools.product(*collapsed_list):
            selector = ''
            for s in t:
                if s[0] == '[':
                    selector += s
                else:
                    selector = selector + '/' + s
            selector_list.append(selector)
        return ','.join(selector_list)

    @classmethod
    def are_disjoint(cls, *selectors):
        """
        Check whether several selectors are disjoint.

        Parameters
        ----------
        s0, s1, ... : Selector, str, unicode, or sequence
            Selectors to check. Each selector is either a
            Selector class instance, a string (e.g.,
            '/foo[0:2]'), or a sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : bool
            True if none of the identifiers comprised by one selector are
            comprised by the other.

        Notes
        -----
        The selectors must not be ambiguous.

        The empty selector is deemed to be disjoint to all other selectors.
        """

        assert len(selectors) >= 1
        assert all(map(cls.is_selector, selectors))
        if len(selectors) == 1: return True
        assert all(map(lambda s: not cls.is_ambiguous(s), selectors))

        # Expand selectors into sets of identifiers:
        ids = set()
        for selector in selectors:

            # Skip empty selectors; they are seemed to be disjoint to all
            # selectors:
            ids_new = set(map(tuple, cls.expand(selector)))
            if ids_new == set([()]):
                continue

            # If some identifiers are present in both the previous expanded
            # selectors and the current selector, the selectors cannot be disjoint:
            if ids.intersection(ids_new):
                return False
            else:
                ids = ids.union(ids_new)
        return True

    @classmethod
    def count_ports(cls, selector):
        """
        Count number of distinct port identifiers in unambigious selector.

        Parameters
        ----------
        selector : Selector, str, unicode, or sequence
            Selector class instance, string (e.g., '/foo[0:2]'),
            or sequence of token sequences (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        count : int
            Number of identifiers comprised by selector.
        """

        e = cls.expand(selector)
        if e == [()] or e == ((),):
            return 0
        else:
            return len(e)

    # Need to create cache here because one can't assign create a cache that is
    # an attribute of the classmethod itself:
    __max_levels_cache = {}
    @classmethod
    def max_levels(cls, selector):
        """
        Return maximum number of token levels in selector.

        Parameters
        ----------
        selector : Selector, str, unicode, or sequence
            Selector class instance, string (e.g., '/foo[0:2]'), or
            sequence of token sequences (e.g., [['foo', slice(0, 2)]]).

        Returns
        -------
        count : int
            Maximum number of tokens in selector.
        """

        assert cls.is_selector(selector)

        # Selector class instances already contain max_levels precomputed:
        if isinstance(selector, Selector):
            return selector.max_levels

        # Handle unhashable selectors:
        try:
            hash(selector)
        except:
            h = _packb(selector)
        else:
            h = selector

        # Use memoization:
        try:
            return cls.__max_levels_cache[h]
        except:
            if isinstance(selector, Selector):
                return selector.max_levels
            elif isinstance(selector, basestring):
                try:
                    count = max(map(len, cls.parse(selector)))
                except:
                    count = 0
            elif type(selector) in [list, tuple]:
                try:
                    count = max(map(len, selector))
                except:
                    count = 0
            else:
                raise ValueError('invalid selector type')
            cls.__max_levels_cache[h] = count
            return count

    @classmethod
    def _multiindex_row_in(cls, row, parse_list, start=None, stop=None):
        """
        Check whether a row in a MultiIndex matches a parsed selector.

        Check whether the entries in a (subinterval of a) given tuple of data
        corresponding to the entries of one row in a MultiIndex match the
        specified token values.

        Parameters
        ----------
        row : sequence
            Data corresponding to a single row of a MultiIndex.
        parse_list : list
            List of lists of token values extracted by ply.
        start, stop : int
            Start and end indices in `row` over which to test entries. If
            the

        Returns
        -------
        result : bool
            True of all entries in specified subinterval of row match,
            False otherwise.
        """

        row_sub = row[start:stop]
        for tokens in parse_list:

            # A single row will never match an empty token list:
            if not tokens:
                continue

            # Check whether all of the entries in `row_sub` match some list of
            # tokens. If this loop terminates prematurely because of a mismatch
            # between `row_sub` and some list of tokens in `parse_list`, it will
            # not return True; this forces checking of the subsequent token
            # lists:
            for i, token in enumerate(tokens):

                # '*' matches everything:
                if token == '*':
                    continue

                # Integers and strings must match exactly:
                elif isinstance(token, (int, long, basestring)):
                    if row_sub[i] != token:
                        break

                # Tokens must be in a set of values:
                elif type(token) == list:
                    if row_sub[i] not in token:
                        break

                # Token must be within range of an interval:
                elif type(token) == slice:
                    i_start = token.start
                    i_stop = token.stop

                    # Handle intervals with ambiguous start or stop values:
                    if (i_start is not None and row_sub[i] < i_start) or \
                       (i_stop is not None and row_sub[i] >= i_stop):
                        break
                else:
                    continue
            else:
                return True

        # If the function still hasn't returned, no match was found:
        return False

    @classmethod
    def _index_row_in(cls, row, parse_list):
        """
        Check whether a row in an Index matches a parsed selector.

        Check whether a row label in an Index instance matches the
        specified token values.

        Parameters
        ----------
        row : scalar
            Data corresponding to a single row of an Index.
        parse_list : list
            List of lists of token values extracted by ply.

        Returns
        -------
        result : bool
            True of all entries in specified subinterval of row match,
            False otherwise.
        """

        # Since `row` is a scalar, it need only match the sole entry of one of
        # the lists in `parse_list`:
        for tokens in parse_list:
            if not tokens:
                continue
            if len(tokens) > 1:
                raise ValueError('index row only is scalar')
            if tokens[0] == '*':
                return True
            elif isinstance(tokens[0], (int, long, basestring)):
                if row == tokens[0]:
                    return True
            elif type(tokens[0]) == list:
                if row in tokens[0]:
                    return True
            elif type(tokens[0]) == slice:
                i_start = tokens[0].start
                i_stop = tokens[0].stop
                if (i_start is None or row >= i_start) and \
                   (i_stop is None or row < i_stop):
                    return True
            else:
                continue
        return False

    @classmethod
    def is_in(cls, s, t):
        """
        Check whether all of the identifiers in one selector are comprised by another.

        Parameters
        ----------
        s, t : Selector, str, unicode, or sequence
            Check whether selector `s` is in `t`. Each selector is either a
            Selector class instance, a string (e.g., '/foo[0:2]'), or a sequence
            of token sequences (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : bool
            True if the first selector is in the second, False otherwise. If `s`
            is an empty selector, this method always returns True.
        """

        assert cls.is_selector(s)
        assert cls.is_selector(t)

        s_exp = set(cls.expand(s))
        if s_exp == set([()]):
            return True
        t_exp = set(cls.expand(t))
        if s_exp.issubset(t_exp):
            return True
        else:
            return False

    @classmethod
    def get_tuples(cls, df, selector, start=None, stop=None):
        """
        Return tuples containing index labels selected by specified selector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : Selector, str, unicode, or sequence
            Selector class instance, string (e.g., '/foo[0:2]'), or sequence
            of token sequences (e.g., [['foo', (0, 2)]]).
        start, stop : int
            Start and end indices in `row` over which to test entries.
            If the index of `df` is an Index, these are ignored.

        Returns
        -------
        result : list
            List of tuples containing index labels for selected rows. If
            `df.index` is an Index, the result is a list of labels.
        """

        assert cls.is_selector(selector)
        max_levels = cls.max_levels(selector)
        if isinstance(selector, Selector):
            parse_list = selector.expanded
        elif isinstance(selector, basestring):
            try:
                parse_list = cls.expand(selector, max_levels)
            except:
                parse_list = cls.parse(selector)
        elif type(selector) in [list, tuple]:
            parse_list = selector
        else:
            raise ValueError('invalid selector type')

        # The maximum number of tokens must not exceed the number of levels in the
        # DataFrame's MultiIndex:
        if max_levels > len(df.index.names[start:stop]):
            raise ValueError('Maximum number of levels in selector exceeds that of '
                             'DataFrame index')

        if isinstance(df.index, pd.MultiIndex):
            return [t for t in df.index \
                    if cls._multiindex_row_in(t, parse_list, start, stop)]
        else:
            return [(t,) for t in df.index \
                    if cls._index_row_in(t, parse_list)]

    @classmethod
    def get_index(cls, df, selector, start=None, stop=None, names=[]):
        """
        Return index corresponding to rows selected by specified selector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : str or unicode
            Row selector.
        start, stop : int
            Start and end indices in `row` over which to test entries.
        names : scalar or list
            Name or names of levels to use in generated index.

        Returns
        -------
        result : pandas.Index or pandas.MultiIndex
            Index that refers to the rows selected by the specified
            selector.
        """

        assert cls.is_selector(selector)

        tuples = cls.get_tuples(df, selector, start, stop)
        if not tuples:
            raise ValueError('no tuples matching selector found')

        # XXX This probably could be made faster by directly manipulating the
        # existing MultiIndex:
        if all(map(np.iterable, tuples)):
            if np.iterable(names) and names:
                return pd.MultiIndex.from_tuples(tuples, names=names)
            elif names:
                return pd.MultiIndex.from_tuples(tuples, names=[names])
            else:
                return pd.MultiIndex.from_tuples(tuples)
        else:
            if np.iterable(names) and names:
                return pd.Index(tuples, name=names[0])
            elif names:
                return pd.Index(tuples, name=names)
            else:
                return pd.Index(tuples)

    @classmethod
    def index_to_selector(cls, idx):
        """
        Convert an index into an expanded port selector.

        Parameters
        ----------
        idx : pandas.Index or pandas.MultiIndex
            Index containing port identifiers.

        Returns
        -------
        selector : list of tuple
            List of tuples corresponding to individual port identifiers.
        """

        if isinstance(idx, pd.MultiIndex):
            return idx.tolist()
        else:
            return [(i,) for i in idx.tolist()]

    @classmethod
    def pad_tuple_list(cls, tuple_list, pad_len):
        """
        Pad a list of tuples with blank strings.

        Parameters
        ----------
        tuple_list : list of tuples
            List of tuples to process.
        pad_len : int
            Length to which tuples should be padded with blanks.

        Returns
        -------
        result : list of tuples
            List of padded tuples.
        """

        return [tuple(x)+('',)*(pad_len-len(x)) for x in tuple_list]

    @classmethod
    def pad_selector(cls, selector, pad_len=float('inf')):
        """
        Expand and pad a selector with blank tokens.

        Expand a selector and pad those port identifier token sequences
        that contain fewer tokens than the specified maximum.

        Parameters
        ----------
        selector : Selector, str, unicode, or sequence
            Selector class instance, string (e.g., '/foo[0:2]'), or sequence
            of token sequences (e.g., [['foo', slice(0, 2)]]).
        pad_len : int
            Length to which expanded token sequences should be padded with blanks.
            If infinite, the sequences are padded to the length of the longest
            sequence.

        Returns
        -------
        padded : sequence
            Sequence of token sequences padded with blank strings.
        """

        if isinstance(selector, Selector):
            expanded = selector.expanded
            max_levels = selector.max_levels
        else:
            expanded = cls.expand(selector)
            max_levels = max(map(len, expanded))

        if pad_len == float('inf'):
            return cls.pad_tuple_list(expanded, max_levels)
        elif pad_len == 0:
            return expanded
        else:
            return cls.pad_tuple_list(expanded, pad_len)

    @classmethod
    def make_index_two_concat(cls, sel_0, sel_1, names=[]):
        """
        Create an index from two selectors concatenated elementwise.

        Parameters
        ----------
        sel_0, sel_1 : str or sequence
            Selector strings (e.g., '/foo[0:2]') or sequence of token
            sequences (e.g., [['foo', (0, 2)]]). Both of the selectors must
            comprise the same number of port identifiers.
        names : list
            Names of levels to use in generated MultiIndex. If no names are
            specified, the levels are assigned increasing integers starting with
            0 as their names.

        Returns
        -------
        result : pandas.MultiIndex
            MultiIndex whose rows are each the concatenation of the
            corresponding rows in `sel_0` and `sel_1`. Each row contains twice
            the maximum number of tokens in the two selectors.

        Notes
        -----
        The selectors may not contain ambiguous symbols such as '*' or '[:]'.
        """

        assert cls.is_selector(sel_0)
        assert not cls.is_ambiguous(sel_0)
        assert cls.is_selector(sel_1)
        assert not cls.is_ambiguous(sel_1)

        sels_0 = cls.expand(sel_0)
        sels_1 = cls.expand(sel_1)

        assert len(sels_0) == len(sels_1)
        N_sel = len(sels_0)

        levels = [[]]
        max_levels_0 = max(map(len, sels_0)) if N_sel else 0
        max_levels_1 = max(map(len, sels_1)) if N_sel else 0
        max_levels = max(max_levels_0, max_levels_1)

        selectors = []
        for i in range(N_sel):

            # Pad expanded selectors:
            sels_0[i] = list(sels_0[i])
            sels_1[i] = list(sels_1[i])

            n = len(sels_0[i])
            if n < max_levels:
                sels_0[i].extend(['' for k in range(max_levels-n)])
            m = len(sels_1[i])
            if m < max_levels:
                sels_1[i].extend(['' for k in range(max_levels-m)])

            # Concatenate:
            selectors.append(sels_0[i]+sels_1[i])

            # Extract level values:
            for k in range(max_levels*2):
                if len(levels) < k+1:
                    levels.append([])
                levels[k].append(selectors[-1][k])

        # Discard duplicate level values:
        levels = [list(set(level)) for level in levels]

        # Start with at least one label so that a valid Index will be returned
        # if the selector is empty:
        labels = [[]]

        # Construct label indices:
        for i in range(N_sel):
            for j in range(max_levels*2):
                if len(labels) < j+1:
                    labels.append([])
                labels[j].append(levels[j].index(selectors[i][j]))

        if not names:
            names = range(len(levels))
        return pd.MultiIndex(levels=levels, codes=labels, names=names)

    @classmethod
    def make_index_two_prod(cls, sel_0, sel_1, names=[]):
        """
        Create an index from the product of two selectors.

        Parameters
        ----------
        sel_0, sel_1 : str or sequence
            Selector strings (e.g., '/foo[0:2]') or sequence of token
            sequences (e.g., [['foo', (0, 2)]]).
        names : list
            Names of levels to use in generated MultiIndex. If no names are
            specified, the levels are assigned increasing integers starting with
            0 as their names.

        Returns
        -------
        result : pandas.MultiIndex
            MultiIndex whose rows are the product of the corresponding rows in
            `sel_0` and `sel_1`. Each row contains twice the maximum number of
            tokens in the two selectors.

        Notes
        -----
        The selectors may not contain ambiguous symbols such as '*' or '[:]'.
        """

        assert cls.is_selector(sel_0)
        assert not cls.is_ambiguous(sel_0)
        assert cls.is_selector(sel_1)
        assert not cls.is_ambiguous(sel_1)

        sels_0 = cls.expand(sel_0)
        sels_1 = cls.expand(sel_1)

        N_sel_0 = len(sels_0)
        N_sel_1 = len(sels_1)

        levels = [[]]
        max_levels_0 = max(map(len, sels_0)) if N_sel_0 else 0
        max_levels_1 = max(map(len, sels_1)) if N_sel_1 else 0
        max_levels = max(max_levels_0, max_levels_1)

        selectors = []
        for i, j in itertools.product(range(N_sel_0), range(N_sel_1)):

            # Pad expanded selectors:
            sels_0[i] = list(sels_0[i])
            sels_1[j] = list(sels_1[j])

            n = len(sels_0[i])
            if n < max_levels:
                sels_0[i].extend(['' for k in range(max_levels-n)])
            m = len(sels_1[j])
            if m < max_levels:
                sels_1[j].extend(['' for k in range(max_levels-m)])

            # Concatenate:
            selectors.append(sels_0[i]+sels_1[j])

            # Extract level values:
            for k in range(max_levels*2):
                if len(levels) < k+1:
                    levels.append([])
                levels[k].append(selectors[-1][k])

        # Discard duplicate level values:
        levels = [list(set(level)) for level in levels]

        # Start with at least one label so that a valid Index will be returned
        # if the selector is empty:
        labels = [[]]

        # Construct label indices:
        N_sel = N_sel_0*N_sel_1
        for i in range(N_sel):
            for j in range(max_levels*2):
                if len(labels) < j+1:
                    labels.append([])
                labels[j].append(levels[j].index(selectors[i][j]))

        if not names:
            names = range(len(levels))
        return pd.MultiIndex(levels=levels, codes=labels, names=names)

    @classmethod
    def make_index(cls, selector, names=[]):
        """
        Create an index from the specified selector.

        Parameters
        ----------
        selector : Selector, str, unicode, or sequence
            Selector class instance, string (e.g., '/foo[0:2]'), or
            sequence of token sequences (e.g., [['foo', (0, 2)]]).
        names : list
            Names of levels to use in generated MultiIndex. If no names are
            specified, the levels are assigned increasing integers starting with
            0 as their names.

        Returns
        -------
        result : pandas.Index or pandas.MultiIndex
            MultiIndex corresponding to the specified selector. If the selector
            only contains a single level, an Index is returned (this is due to a
            pecularity of pandas).

        Notes
        -----
        The selector may not contain ambiguous symbols such as '*' or '[:]'.
        """

        assert cls.is_selector(selector)
        assert not cls.is_ambiguous(selector)

        # Since nonempty Selectors are already expanded into lists of tuples,
        # MultiIndex.from_tuples() can be used directly:
        if isinstance(selector, Selector) and selector.nonempty:
            if not names:
                names = range(selector.max_levels)
            return pd.MultiIndex.from_tuples(selector.expanded,
                                             names=names)

        # XXX It might be preferable to make expand()
        # convert all output to a tuple rather than just doing so here:
        selectors = tuple(cls.expand(selector))

        N_sel = len(selectors)
        sel_lens =  list(map(len, selectors))
        max_levels = max(sel_lens) if N_sel else 0

        # NaNs in index are not supported by MultiIndex. Create from tuples
        # only if all selectors have same levels.
        if len(set(sel_lens)) == 1:
            if not names:
                if max_levels:
                    names = range(max_levels)
                else:
                    names = [0]

            if selectors == ((),):
                return pd.MultiIndex(levels=[[] for _ in range(len(names))],
                                     codes=[[] for _ in range(len(names))],
                                     names=names)
            else:
                return pd.MultiIndex.from_tuples(selectors, names=names)

        # Accumulate unique values for each level of the MultiIndex:
        levels = [set() for i in range(max_levels)]
        for i in range(N_sel):
            for j in range(sel_lens[i]):
                levels[j].add(selectors[i][j])
            for j in range(sel_lens[i], max_levels):
                levels[j].add('')

        # Sort levels:
        levels = [list(level) for level in levels]

        # Construct label indices:
        labels = [[] for i in range(max_levels)]
        for i in range(N_sel):
            for j in range(sel_lens[i]):
                labels[j].append(levels[j].index(selectors[i][j]))
            for j in range(sel_lens[i], max_levels):
                labels[j].append(levels[j].index(''))

        if not names:
            names = range(len(levels))
        return pd.MultiIndex(levels=levels, codes=labels, names=names)

    @classmethod
    def select(cls, df, selector, start=None, stop=None):
        """
        Select rows from DataFrame using a path-like selector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : Selector, str, unicode, or sequence
            Selector class instance, string (e.g., '/foo[0:2]') or
            sequence of token sequences (e.g., [['foo', (0, 2)]]).
        start, stop : int
            Start and end indices in `row` over which to test entries.

        Returns
        -------
        result : pandas.DataFrame
            DataFrame containing selected rows.
        """

        assert cls.is_selector(selector)
        if isinstance(selector, Selector):
            if len(df.index.names[start:stop])>1:
                try:
                    tks = list(selector.expanded)
                    return df.loc[tks]
                except:
                    pass
            parse_list = list(selector.expanded)
        elif isinstance(selector, basestring):
            if len(df.index.names[start:stop])>1:
                try:
                    tks = cls.expand(selector)
                    return df.loc[tks]
                except:
                    pass
            parse_list = cls.parse(selector)
        elif type(selector) in [list, tuple]:
            try:
                tks = cls.expand(selector)
                return df.loc[tks]
            except:
                pass
            parse_list = selector
        else:
            raise ValueError('invalid selector type')

        # The number of tokens must not exceed the number of levels in the
        # DataFrame's MultiIndex; the maximum number of levels in a selector
        # containing no identifiers is obviously 0:
        max_levels = max(map(len, parse_list)) if len(parse_list) else 0
        if max_levels > len(df.index.names[start:stop]):
            raise ValueError('Number of levels in selector exceeds number in row subinterval')

        if type(df.index) == pd.MultiIndex:
            # deprecated in pandas
            # return df.select(lambda row: cls._multiindex_row_in(row, parse_list,
            #                                                     start, stop))
            return df.loc[df.index.map(lambda row: cls._multiindex_row_in(row, parse_list,
                                                                start, stop))]
        else:
            # deprecated in pandas
            # return df.select(lambda row: cls._index_row_in(row, parse_list))
            return df.loc[df.index.map(lambda row: cls._index_row_in(row, parse_list))]

# Set the option optimize=1 in the production version; need to perform these
# assignments after definition of the rest of the class because the class'
# internal namespace can't be accessed within its body definition:
optimize = 0
SelectorParser.lexer = lex.lex(module=SelectorParser, optimize=optimize)
SelectorParser.parser = yacc.yacc(module=SelectorParser,
                                  debug=0, write_tables=0, optimize=optimize)
