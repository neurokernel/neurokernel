#!/usr/bin/env python

"""
Represent connectivity pattern using pandas DataFrame.
"""

import itertools
import numpy as np
import pandas as pd

from plsel import PathLikeSelector

class Pattern(object):
    """
    Class for representing connectivity between sets of interface ports.

    This class represents connection mappings from one set of ports to another.
    Ports are represented using path-like identifiers as follows:

    p = Pattern()
    p['/x[0:3]', '/y[0:2]'] = 1

    A single attribute ('conn') is created by default. Specific attributes
    may be accessed by specifying their names after the port identifiers; if
    a nonexistent attribute is specified when a sequential value is assigned,
    a new column for that attribute is automatically created:

    p['/x[0:3]', '/y[0:2', 'conn', 'x'] = [1, 'foo']

    Parameters
    ----------
    data : numpy.ndarray, dict, or pandas.DataFrame
        Data to load store in class instance.
    from_sel, to_sel : str
        Selectors that describe the pattern's initial index. If specified, 
        both selectors must be set. If no selectors are set, the index is
        initially empty.
    columns : sequence of str
        Data column names.
    """

    def __init__(self, data=None, from_sel=None, to_sel=None, columns=['conn']):
        self.sel = PathLikeSelector()

        if (from_sel is None and to_sel is None):
            from_levels = 1
            to_levels = 1
        elif (from_sel is not None and to_sel is not None):
            from_levels = self.sel.max_levels(from_sel)
            to_levels = self.sel.max_levels(to_sel)
        else:
            raise ValueError('Both from and to selectors must either be set or unset')
            
        self.num_levels = {'from': from_levels, 'to': to_levels}
        names = ['from_%s' % i for i in xrange(self.num_levels['from'])]+ \
                ['to_%s' %i for i in xrange(self.num_levels['to'])]

        # Construct index from concatenated selectors if specified:
        if (from_sel is None and to_sel is None):
            levels = [[]]*len(names)
            labels = [[]]*len(names)
            idx = pd.MultiIndex(levels=levels, labels=labels, names=names)
        else:
            idx = self.sel.make_index('(%s)+(%s)' % (from_sel, to_sel), names)
        self.data = pd.DataFrame(data=data, columns=columns, index=idx)

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
        
    def __setitem__(self, key, value):

        # Check whether the number of levels in the internal DataFrame's
        # MultiIndex must be increased to accommodate the specified selector:
        for i in xrange(self.sel.max_levels(key[0])-self.num_levels['from']):
            self.__add_level__('from')
        for i in xrange(self.sel.max_levels(key[1])-self.num_levels['to']):
            self.__add_level__('to')

        # Try using the selector to select data from the internal DataFrame:
        selector = '+'.join(key[0:2])
        try:
            idx = self.sel.get_index(self.data, selector,
                                     names=self.data.index.names)
        
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
        if len(key) > 2:
            if np.isscalar(value):
                data = {k:value for k in key[2:]}
            elif type(value) == dict:
                data = value
            elif np.iterable(value) and len(value) <= len(key[2:]):
                data={k:v for k, v in zip(key[2:], value)}
            else:
                raise ValueError('cannot assign specified value')
        else:
            if np.isscalar(value):
                data = {self.data.columns[0]: value}
            elif type(value) == dict:
                data = value
            elif np.iterable(value) and len(value) <= len(self.data.columns):
                data={k:v for k, v in zip(self.data.columns, value)}
            else:
                raise ValueError('cannot assign specified value')

        if found:
            for k, v in data.iteritems():
                self.data[k].ix[idx] = v
        else:
            self.data = self.data.append(pd.DataFrame(data=data, index=idx))
            self.data.sort(inplace=True)

    def __getitem__(self, key):
        if len(key) > 2:
            return self.sel.select(self.data[list(key[2:])],
                                             selector = '+'.join(key[0:2]))
        else:
            return self.sel.select(self.data, selector = '+'.join(key))

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

        def test_create_no_init_idx(self):
            c = Pattern(columns=['conn','from_type', 'to_type'])
            c['/foo[0]', '/bar[0]'] = [1, 'spike', 'spike']
            c['/foo[0]', '/bar[1]'] = [1, 'spike', 'spike']
            c['/foo[2]', '/bar[2]'] = [1, 'spike', 'spike']
            c['/bar[0]', '/foo[0]'] = [1, 'gpot', 'gpot']
            c['/bar[1]', '/foo[0]'] = [1, 'gpot', 'gpot']
            c['/bar[2]', '/foo[1]'] = [1, 'spike', 'gpot']
            assert_frame_equal(c.data, self.df)

    main()
