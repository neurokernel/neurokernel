#!/usr/bin/env python

"""
Experiment with using xpath-like "keys" to access pandas data.
"""

import re
import numpy as np
import pandas as pd

N = 10
df1 = pd.DataFrame(data={'data': np.random.rand(N),
                        'key': ['/foo[0]',
                                '/foo[1]',
                                '/foo[2]',
                                '/foo[3]',
                                '/foo[4]',
                                '/bar[0]',
                                '/bar[1]',
                                '/bar[2]',
                                '/baz[0]',
                                '/baz[1]']})
print df1[df1.key.str.contains('/foo.*')]
print df1[df1.key.str.contains('/bar.*')]
print df1[df1.key.str.contains('/.*[0]')]

df2 = pd.DataFrame(data={'data': np.random.rand(N), 
                         0: ['foo', 'foo', 'foo', 'foo', 'foo', 
                             'bar', 'bar', 'bar', 
                             'baz', 'baz'], 
                         1: [0, 1, 2, 3, 4, 0, 1, 2, 0, 1]})
df2.set_index(0, append=False, inplace=True)
df2.set_index(1, append=True, inplace=True)

df3 = pd.DataFrame(data={'data': np.random.rand(N),
                         0: ['foo', 'foo', 'foo', 'foo', 'foo', 
                             'bar', 'bar', 'bar', 
                             'baz', 'baz'],
                         1: ['qux', 'qux', 'mof', 'mof', 'mof',
                             'qux', 'qux', 'qux', 'qux', 'mof'],
                         2: [0, 1, 0, 1, 2, 0, 1, 2, 0, 0]})
df3.set_index(0, append=False, inplace=True)
df3.set_index(1, append=True, inplace=True)
df3.set_index(2, append=True, inplace=True)

import re
import ply.lex as lex

class XPathSelector(object):
    """
    Class for selecting rows of a pandas DataFrame using XPATH-like selectors. 
    Assumes that the DataFrame instance has a MultiIndex where each level
    corresponds to part of the selector.

    Examples
    --------
    /foo
    /foo/bar
    /foo/bar[0]
    /foo/bar[0:5]
    /foo/*/baz
    /foo/*/baz[5]
    """

    tokens = ('LEVEL',)

    def __init__(self):
        self.build()

    def t_LEVEL(self, t):
        r'(?:/[^/\[\]:]+)|(?:\[(?:\d+)\])|(?:\[\d*\:\d*\])'
        if re.search('[\[\]]', t.value):
            t.value = re.search('\[(.+)\]', t.value).group(1)
        else:
            t.value = t.value.strip('/')
        return t

    def t_error(self, t):
        print 'Illegal character "%s"' % t.value[0]
        raise ValueError('Cannot parse selector')

    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def parse(self, data):
        self.lexer.input(data)
        token_list = []
        while True:
            tok = self.lexer.token()
            if not tok: break
            token_list.append(tok)
        return token_list

    def _slice_str_to_tuple(self, slice_str):
        """
        Convert string representation of slice to interval containing numerical
        start and stop values.
        """
        start, stop = slice_str.split(':')
        if start == '':
            start = 0
        else:
            start = int(start)
        if stop == '':
            stop = np.inf
        else:
            stop = int(stop)
        return (start, stop)

    def select(self, df, data):
        token_list = self.parse(data)

        # The number of levels must be equivalent to the number of levels in the
        # DataFrame's MultiIndex:        
        if len(token_list) > len(df.index.names):
            raise ValueError('Number of levels in selector exceeds that of '
                             'DataFrame index')
        def select_func(row):
            for i, x in enumerate(row):
                if token_list[i] == '*':
                    continue
                elif re.match('\d*\:\d*', token_list[i]):
                    start, stop = self._slice_str_to_tuple(token_list[i])

m = XPathSelector()
m.parse('/foo/*[0:10]/bar[0:]/*[0]')

