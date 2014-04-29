#!/usr/bin/env python

"""
Represent connectivity pattern using pandas DataFrame.
"""

import itertools
import numpy as np
import pandas as pd

from plsel import PathLikeSelector

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

    This class represents connection mappings from one set of ports to another.
    Ports are represented using path-like identifiers.

    Parameters
    ----------
    from_levels, to_levels : int
        Maximum mumber of levels in the source and destination port selectors.
    """

    def __init__(self, from_levels=1, to_levels=1):        
        self.sel = PathLikeSelector()

        self.num_levels = {'from': from_levels, 'to': to_levels}
        names = ['from_%s' % i for i in xrange(self.num_levels['from'])]+ \
                ['to_%s' %i for i in xrange(self.num_levels['to'])]
        levels = [[]]*len(names)
        labels = [[]]*len(names)
        idx = pd.MultiIndex(levels=levels, labels=labels, names=names)
        self.data = pd.DataFrame(columns=['conn', 'from_type', 'to_type'], index=idx)

    def __add_level__(self, which):
        assert which in ['from', 'to']

        # Check whether the level corresponds to the 'from' or 'to' part of the
        # connectivity pattern:
        new_level_name = '%s_%s' % (which, self.num_levels[which])

        # Add a data column corresponding to the new level:
        self.data[new_level_name] = ''

        # Convert to MultiIndex level:
        self.data.set_index(new_level_name, append=True, inplace=True)

        # Rearrange the MultiIndex levels so that the 'from' and 'to' levels
        # remain grouped together: 
        if which == 'from':
            order = range(self.num_levels['from'])+\
                    [self.num_levels['from']+self.num_levels['to']]+\
                    range(self.num_levels['from'], self.num_levels['from']+self.num_levels['to'])
            self.data.index = self.data.index.reorder_levels(order)

        # Bump number of levels:
        self.num_levels[which] += 1

    def __key_to_selector__(self, key):
        """
        Convert a key consisting of a tuple of strings into a single selector string.
        """

        # Validate input:
        if len(key) != 2 or type(key[0]) != str or type(key[1]) != str:
            raise ValueError('invalid key')

        # If the number of tokens in the first selector is less than the number
        # of levels in the internal DataFrame, pad it with '*' before combining with
        # the second selector:
        num_toks_0 = self.sel.count_tokens(key[0])
        pad = ''.join(['/*']*(self.num_levels['from']-num_toks_0))
        return key[0]+pad+key[1]
        
    def __setitem__(self, key, value):

        # Check whether the number of levels in the internal DataFrame's
        # MultiIndex must be increased to accommodate the specified selector:
        for i in xrange(self.sel.count_tokens(key[0])-self.num_levels['from']):
            self.__add_level__('from')
        for i in xrange(self.sel.count_tokens(key[1])-self.num_levels['to']):
            self.__add_level__('to')

        # Try using the selector to select data from the internal DataFrame:
        selector = self.__key_to_selector__(key)
        try:
            idx = self.sel.get_index(self.data, selector, names=self.data.index.names)
        
        # If the select fails, try to create new rows with the index specified
        # by the selector and load them with the specified data:
        except:
            try:
                idx = self.sel.make_index(selector, self.data.index.names)
            except:
                raise ValueError('cannot create new rows for ambiguous selector %s' % selector)
            else:
                found = False
        else:
            found = True

        # Ensure that data to set is in dict form:
        if np.isscalar(value):
            data = {self.data.columns[0]: value}
        elif type(value) == dict:
            data = value
        elif isiterable(value) and len(value) <= len(self.data.columns):
            data={k:v for k, v in zip(self.data.columns, value)}        

        if found:
            for k, v in data.iteritems():
                self.data[k].ix[idx] = v
        else:
            self.data = self.data.append(pd.DataFrame(data=data, index=idx))
            self.data.sort(inplace=True)

    def __getitem__(self, key):
        selector = self.__key_to_selector__(key)
        return self.sel.select(self.data, selector)

    def __repr__(self):
        return self.data.__repr__()

    def clear(self):
        """
        Clear all entries in class instance.
        """

        self.data.dropna(inplace=True)

    def from_csv(self, file_name, **kwargs):
        """
        Read connectivity data from CSV file.

        Given N 'from' levels and M 'to' levels in the internal index, 
        the method assumes that the first N+M columns in the file specify
        the index levels.

        See Also
        --------
        pandas.read_csv
        """
        
        data_names = self.data.columns
        index_names = self.data.index.names
        kwargs['names'] = data_names
        kwargs['index_col'] = range(len(index_names))
        data = pd.read_csv(file_name, **kwargs)
        self.data = data

        # Restore MultiIndex level names:
        self.data.index.names = index_names

if __name__ == '__main__':
    from unittest import main, TestCase
    from pandas.util.testing import assert_frame_equal

    class test_connectivity(TestCase):
        def setUp(self):
            self.df = pd.DataFrame(data={'conn': np.ones(6, dtype=object),
                            'from_type': ['spike', 'spike', 'spike',
                                          'gpot', 'gpot', 'spike'],
                            'to_type': ['spike', 'spike', 'spike',
                                        'gpot', 'gpot', 'gpot'],
                            'from_0': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
                            'from_1': [0, 0, 2, 0, 1, 2],
                            'to_0': ['bar', 'bar', 'bar', 'foo', 'foo', 'foo'],
                            'to_1': [0, 1, 2, 0, 0, 1]})
            self.df.set_index('from_0', append=False, inplace=True)
            self.df.set_index('from_1', append=True, inplace=True)
            self.df.set_index('to_0', append=True, inplace=True)
            self.df.set_index('to_1', append=True, inplace=True)
            self.df.sort(inplace=True)

        def test_create_conn(self):
            c = Connectivity(2, 2)
            c['/foo[0]', '/bar[0]'] = [1, 'spike', 'spike']
            c['/foo[0]', '/bar[1]'] = [1, 'spike', 'spike']
            c['/foo[2]', '/bar[2]'] = [1, 'spike', 'spike']
            c['/bar[0]', '/foo[0]'] = [1, 'gpot', 'gpot']
            c['/bar[1]', '/foo[0]'] = [1, 'gpot', 'gpot']
            c['/bar[2]', '/foo[1]'] = [1, 'spike', 'gpot']
            assert_frame_equal(c.data, self.df)

    main()
