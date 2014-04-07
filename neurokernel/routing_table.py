#!/usr/bin/env python

"""
Routing table class.
"""

import numpy as np

import pandas as pd

class RoutingTable(object):
    """
    Routing table class.

    Simple class that stores pairs of strings that can signify
    one-hop routes between entities in a graph. Assigning a value to a
    pair that isn't in the class instance will result in the pair and value
    being appended to the class instance.

    Parameters
    ----------
    df : pandas.DataFrame
        Initial table of routing data. The dataframe must have a MultiIndex
        containing two levels `i` and `j`.

    Notes
    -----
    Assigning to a new pair will result in reallocation of the internal
    dataframe. This shouldn't be a big concern given that expected maximum
    number of distinct entities in the table shouldn't be more than 50.
    """
    
    def __init__(self, df=None):
        if df is None:            
            self.data = pd.DataFrame(columns=['data'],
                                      index=pd.MultiIndex(levels=[[], []],
                                                          labels=[[], []],
                                                          names=['i', 'j']))
        else:
            try:
                type(df) == pd.DataFrame
                len(df.index.levels) == 2
                len(df.index.labels) == 2
            except:
                raise ValueError('invalid initial array')
            else:
                self.data = df.copy()

    def __setitem__(self, key, value):
        if type(key) == slice:
            raise ValueError('assignment by slice not supported')
        if len(key) != 2:
            raise KeyError('invalid key')
        try:
            self.data.ix[key]
        except:
            self.data = self.data.append(pd.DataFrame({'data': value},
                                                        index=pd.MultiIndex.from_tuples([key])))
            self.data.sort(inplace=True)
        else:
            self.data.ix[key] = value

    def __getitem__(self, key):
        raise NotImplementedError

    def __copy__(self):
        return self.__class__(self.data)

    copy = __copy__

    @property
    def shape(self):
        """
        Shape of table.
        """

        return len(self.data.index.levels[0]), len(self.data.index.levels[1])

    @property
    def ids(self):
        """
        IDs currently in routing table.
        """

        return list(self.data.index.levels[0]+self.data.index.levels[1])

    @property
    def coords(self):
        """
        List of coordinate tuples of all nonzero table entries.
        """

        return list(self.data[self.data['data'] != 0].index)

    @property
    def nnz(self):
        """
        Number of nonzero entries in the table.
        """

        return len(self.data[self.data['data'] != 0])
    
    def row_ids(self, col_id):
        """
        Row IDs connected to a column ID.
        """

        try:
            return list(self.data[self.data['data'] != 0].xs(col_id, level='j').index)
        except:
            return []

    def all_row_ids(self):
        """
        All row IDs connected to column IDs.
        """
        
        d = self.data[self.data['data'] != 0]
        return list(set(d.index.levels[0][d.index.labels[0]]))

    def col_ids(self, col_id):
        """
        Column IDs connected to a row ID.
        """

        try:
            return list(self.data[self.data['data'] != 0].ix[col_id].index)
        except:
            return []

    def all_col_ids(self):
        """
        All column IDs connected to row IDs.
        """
        
        d = self.data[self.data['data'] != 0]
        return list(set(d.index.levels[1][d.index.labels[1]]))

    def __repr__(self):
        return self.data.__repr__()

if __name__ == '__main__':
    from unittest import main, TestCase

    class test_routingtable(TestCase):
        def setUp(self):
            self.coords_orig = [('a', 'b'), ('b', 'c')]
            self.ids_orig = set([i[0] for i in self.coords_orig]+\
                                [i[1] for i in self.coords_orig])            
            self.t = RoutingTable()
            for c in self.coords_orig:
                self.t[c[0], c[1]] = 1
            self.shape = (2, 2)
        def test_shape(self):
            assert self.t.shape == self.shape
        def test_ids(self):
            assert set(self.t.ids) == self.ids_orig
        def test_coords(self):
            assert set(self.t.coords) == set(self.coords_orig)
        def test_nnz(self):
            assert self.t.nnz == len(self.coords_orig)
        def test_all_row_ids(self):
            assert set(self.t.all_row_ids()) == \
                set([i[0] for i in self.coords_orig])
        def test_all_col_ids(self):
            assert set(self.t.all_col_ids()) == \
                set([i[1] for i in self.coords_orig])
        def test_row_ids(self):
            for i in self.coords_orig:
                assert i[0] in self.t.row_ids(i[1])
        def test_col_ids(self):
            for i in self.coords_orig:
                assert i[1] in self.t.col_ids(i[0])
    main()

