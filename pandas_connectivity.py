#!/usr/bin/env python

"""
Represent connectivity pattern using pandas DataFrame.
"""

import itertools
import numpy as np
import pandas as pd

from pandas_xpath_like_keys import XPathSelector

def isiterable(x):
    try:
        iter(x)
    except:
        return False
    else:
        return True

class Connectivity(object):
    """
    Class for representing connectivity between sets of interface ports.
    """

    def __init__(self):
        self.sel = XPathSelector()

        self.from_levels = 1
        self.to_levels = 1
        idx = pd.MultiIndex(levels=[[], []],
                            labels=[[], []],
                            names=['from_%s' % (self.from_levels-1), 
                                   'to_%s' % (self.to_levels-1)])
        self.data = pd.DataFrame(columns=['conn', 'io', 'type'], index=idx)

    def __add_level(self, which):
        if which == 'from':
            pass
        elif which == 'to':
            pass
    def __setitem__(self, key, value):
        """
        Notes
        -----
        conn['/a/port[0]', '/b/port[0]']['conn'] = 1
        """

        # The key must be a tuple of strings:
        if len(key) != 2 or (type(k[0]) != str and type(k[1]) != str):
            raise ValueError('invalid key')

        # If the number of tokens in the first selector is less than the number
        # of levels in the internal DataFrame, pad it with '*' before combining with
        # the second selector:
        num_toks_0 = self.sel.parse(key[0])
        pad = ''.join(['/*']*(self.from_levels-num_toks_0))
        selector = key[0]+pad+key[1]

        # Try using the selector to select data from the internal DataFrame:
        try:
            idx = self.sel.get_index(self.data, selector)
            
        # If the select fails, try to create new rows with the index specified
        # by the selector and load them with the specified data:
        except:
            try:
                idx = self.sel.make_index(selector)
            except:
                raise ValueError('cannot create new rows for ambiguous selector')
            
            if np.isscalar(value):
                data = {self.data.columns[0]: value}
            elif type(value) == dict:
                data = value
            elif isiterable(value) and len(value) <= len(self.data.columns):
                data={k:v for k, v in zip(self.data.columns, value)}
            self.data.append(pd.DataFrame(data=data, index=idx))
            self.data.sort(inplace=True)

        # If the select succeeds, set the corresponding values in the DataFrame:
        else:
            self.data['conn'].ix[idx] = value

    def __getitem__(self, key):
        return self.data.ix[self.sel.select(self.data, key)]

    def __repr__(self):
        return self.data.__repr__()

df = pd.DataFrame(data={'conn': np.ones(6),
                        'from_0': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
                        'from_1': [0, 0, 2, 0, 1, 2],
                        'to_0': ['bar', 'bar', 'bar', 'foo', 'foo', 'foo'],
                        'to_1': [0, 1, 2, 0, 1, 2]})
df.set_index('from_0', append=False, inplace=True)
df.set_index('from_1', append=True, inplace=True)
df.set_index('to_0', append=True, inplace=True)
df.set_index('to_1', append=True, inplace=True)
