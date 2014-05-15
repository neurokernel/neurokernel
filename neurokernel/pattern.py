#!/usr/bin/env python

"""
Represent connectivity pattern using pandas DataFrame.
"""

import itertools
import numpy as np
import pandas as pd

from plsel import PathLikeSelector

class Interface(object):
    """
    Interface comprising ports.

    This class contains information about a set of path-like identifiers [1]_
    and the (optional) attributes associated with them. By default, each port
    has an 'io' attribute (indicating whether it receives
    input or emits output) and a 'type' attribute (indicating whether it
    emits/receives spikes or graded potential values.

    Examples
    --------
    >>> i = Interface('/foo[0:5],/bar[0:3]')
    >>> i['/foo[0]', 'io', 'type'] = ['in', 'spike']

    Attributes
    ----------
    data : pandas.DataFrame
        Port attribute data.
    index : pandas.MultiIndex
        Index of port identifiers.

    Parameters
    ----------
    selector : str
        Selector describing the port identifiers comprised by the interface.
    columns : list
        Data column names.

    See Also
    --------
    .. [1] PathLikeSelector
    """

    def __init__(self, selector, columns=['io', 'type']):
        self.sel = PathLikeSelector()
        assert not(self.sel.isambiguous(selector))        
        self.num_levels = self.sel.max_levels(selector)
        names = [str(i) for i in xrange(self.num_levels)]
        idx = self.sel.make_index(selector, names)
        self.data = pd.DataFrame(index=idx, columns=columns)

    @property
    def index(self):
        """
        Interface index.
        """

        return self.data.index
    @index.setter
    def index(self, i):
        self.data.index = i

    def __add_level__(self):
        """
        Add an additional level to the index of the pattern's internal
        DataFrame.
        """

        # Check whether the level corresponds to the 'from' or 'to' part of the
        # connectivity pattern:
        new_level_name = str(self.num_levels)

        # Add a data column corresponding to the new level:
        self.data[new_level_name] = ''

        # Convert to MultiIndex level:
        self.data.set_index(new_level_name, append=True, inplace=True)

        # Bump number of levels:
        self.num_levels[which] += 1
        
    def __getitem__(self, key):
        return self.sel.select(self.data, key)
        
    def __setitem__(self, key, value):
        if type(key) == tuple:
            selector = key[0]
        else:
            selector = key

        # Check whether the number of levels in the internal DataFrame's
        # MultiIndex must be increased to accommodate the specified selector:
        for i in xrange(self.sel.max_levels(selector)-self.num_levels):
            self.__add_level__()

        # Try using the selector to select data from the internal DataFrame:
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
        if type(key) == tuple and len(key) > 1:
            if np.isscalar(value):
                data = {k:value for k in key[1:]}
            elif type(value) == dict:
                data = value
            elif np.iterable(value) and len(value) <= len(key[1:]):
                data={k:v for k, v in zip(key[1:], value)}
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

    def to_tuples(self):
        """
        Return list of port identifiers as tuples.
        """

        return self.data.index.tolist()

    def to_selectors(self):
        """
        Return list of port identifiers as path-like selectors.
        """

        result = []
        for t in self.to_tuples():
            selector = ''
            for s in t:
                if type(s) == str:
                    selector += '/'+s
                else:
                    selector += '[%s]' % s
            result.append(selector)
        return result

    def __repr__(self):
        return 'Interface\n---------\n'+self.data.__repr__()

class Pattern(object):
    """
    Connectivity pattern linking sets of interface ports.

    This class represents connection mappings between interfaces comprising sets 
    of ports. Ports are represented using path-like identifiers [1]_.
    A single data attribute ('conn') associated with defined connections 
    is created by default. Specific attributes may be accessed by specifying 
    their names after the port identifiers; if a nonexistent attribute is 
    specified when a sequential value is assigned, a new column for that 
    attribute is automatically created: ::

        p['/x[0:3]', '/y[0:2]', 'conn', 'x'] = [1, 'foo']

    Examples
    --------
    >>> p = Pattern('/x[0:3]','/y[0:2]')
    >>> p['/x[0:2]', '/y[0]'] = 1
    >>> p['/y[0:2]', '/x[1]'] = 1

    Attributes
    ----------
    data : pandas.DataFrame
        Connection attribute data.
    index : pandas.MultiIndex
        Index of connections.
    interface : dict of Interface
        Interfaces containing port identifiers and attributes.

    Parameters
    ----------
    sel0, sel1, ...: str
        Selectors defining the sets of ports potentially connected by the 
        pattern. These selectors must be disjoint, i.e., no identifier comprised
        by one selector may be in any other selector.
    columns : sequence of str
        Data column names.

    See Also
    --------
    .. [1] PathLikeSelector
    """

    def __init__(self, *selectors, **kwargs):
        columns = kwargs['columns'] if kwargs.has_key('columns') else ['conn']
        self.sel = PathLikeSelector()

        # Force sets of identifiers to be disjoint so that no identifier can
        # denote a port in more than one set:
        assert self.sel.aredisjoint(selectors)

        # Create and save Interface instances containing the ports 
        # comprised by each of the respective selectors:
        self.interface = {}
        max_levels = 0
        for i, s in enumerate(selectors):
            self.interface[i] = Interface(s)

            # Find the maximum number of levels required to accommodate all of 
            # the identifiers:
            if self.interface[i].num_levels > max_levels:
                max_levels = self.interface[i].num_levels

        # Create a MultiIndex that can store mappings between identifiers in the
        # two interfaces:
        self.num_levels = {'from': max_levels, 'to': max_levels}
        names = ['from_%s' % i for i in xrange(self.num_levels['from'])]+ \
                ['to_%s' %i for i in xrange(self.num_levels['to'])]
        levels = [[] for i in xrange(len(names))]
        labels = [[] for i in xrange(len(names))]
        idx = pd.MultiIndex(levels=levels, labels=labels, names=names)
                            
        self.data = pd.DataFrame(index=idx, columns=columns)

    @property
    def index(self):
        """
        Pattern index.
        """

        return self.data.index
    @index.setter
    def index(self, i):
        self.data.index = i
    @classmethod
    def _create_from(cls, *selectors, **kwargs):
        """
        Create a Pattern instance from the specified selectors.

        Parameters
        ----------
        sel0, sel1, ...: str
            Selectors defining the sets of ports potentially connected by the 
            pattern. These selectors must be disjoint, i.e., no identifier comprised
            by one selector may be in any other selector.   
        from_sel, to_sel : str
            Selectors that describe the pattern's initial index. If specified, 
            both selectors must be set. If no selectors are set, the index is
            initially empty.
        data : numpy.ndarray, dict, or pandas.DataFrame
            Data to load store in class instance.
        columns : sequence of str
            Data column names.
        comp_op : str
            Operator to use to combine selectors into single selector that
            comprises both the source and destination ports in a pattern.
        
        Returns
        -------
        result : Pattern
            Pattern instance.
        """

        from_sel = kwargs['from_sel'] if kwargs.has_key('from_sel') else None
        to_sel = kwargs['to_sel'] if kwargs.has_key('to_sel') else None
        data = kwargs['data'] if kwargs.has_key('data') else None
        columns = kwargs['columns'] if kwargs.has_key('columns') else ['conn']
        comb_op = kwargs['comb_op'] if kwargs.has_key('comb_op') else '+'

        # Create empty pattern:
        p = cls(*selectors, columns=columns)

        # Construct index from concatenated selectors if specified:
        names = p.data.index.names
        if (from_sel is None and to_sel is None):
            levels = [[] for i in xrange(len(names))]
            labels = [[] for i in xrange(len(names))]
            idx = pd.MultiIndex(levels=levels, labels=labels, names=names)
        else:
            idx = p.sel.make_index('(%s)%s(%s)' % (from_sel, comb_op, to_sel), names)
                                   
        # Replace the pattern's DataFrame:
        p.data = pd.DataFrame(data=data, index=idx, columns=columns)
        return p

    @classmethod
    def from_product(cls, *selectors, **kwargs):
        """
        Create pattern from the product of identifiers comprised by two selectors.

        For example: ::

            p = Pattern.from_product('/foo[0:2]', '/bar[0:2]',
                                    from_sel='/foo[0:2]', to_sel='/bar[0:2]',
                                    data=1)

        results in a pattern with the following connections: ::

            '/foo[0]' -> '/bar[0]'
            '/foo[0]' -> '/bar[1]'
            '/foo[1]' -> '/bar[0]'
            '/foo[1]' -> '/bar[1]'

        Parameters
        ----------
        sel0, sel1, ...: str
            Selectors defining the sets of ports potentially connected by the 
            pattern. These selectors must be disjoint, i.e., no identifier comprised
            by one selector may be in any other selector.   
        from_sel, to_sel : str
            Selectors that describe the pattern's initial index.
        data : numpy.ndarray, dict, or pandas.DataFrame
            Data to load store in class instance.
        columns : sequence of str
            Data column names.

        Returns
        -------
        result : Pattern
            Pattern instance.
        """

        from_sel = kwargs['from_sel'] if kwargs.has_key('from_sel') else None
        to_sel = kwargs['to_sel'] if kwargs.has_key('to_sel') else None
        data = kwargs['data'] if kwargs.has_key('data') else None
        columns = kwargs['columns'] if kwargs.has_key('columns') else ['conn']
        return cls._create_from(*selectors, from_sel=from_sel, to_sel=to_sel, 
                                data=data, columns=columns, comb_op='+')

    @classmethod
    def from_concat(cls, *selectors, **kwargs):
        """
        Create pattern from the concatenation of identifers comprised by two selectors.

        For example: ::

            p = Pattern.from_concat('/foo[0:2]', '/bar[0:2]',
                                    from_sel='/foo[0:2]', to_sel='/bar[0:2]',
                                    data=1)

        results in a pattern with the following connections: ::

            '/foo[0]' -> '/bar[0]'
            '/foo[1]' -> '/bar[1]'

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

        Returns
        -------
        result : Pattern
            Pattern instance.
        """

        from_sel = kwargs['from_sel'] if kwargs.has_key('from_sel') else None
        to_sel = kwargs['to_sel'] if kwargs.has_key('to_sel') else None
        data = kwargs['data'] if kwargs.has_key('data') else None
        columns = kwargs['columns'] if kwargs.has_key('columns') else ['conn']
        return cls._create_from(*selectors, from_sel=from_sel, to_sel=to_sel, 
                                data=data, columns=columns, comb_op='.+')

    def __add_level__(self, which):
        """
        Add an additional level to the index of the pattern's internal
        DataFrame.

        Parameters
        ----------
        which : {'from', 'to'}
            Which portion of the index to modify.
        """

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
        
    def ininterfaces(self, selector):
        """
        Check whether a selector is supported by any of the pattern's
        interfaces.
        """

        for interface in self.interface.values():
            if len(interface[selector]) > 0:
                return True
        return False

    def __setitem__(self, key, value):

        # Must pass more than one argument to the [] operators:
        assert type(key) == tuple

        # Ensure that specified selectors refer to ports in the
        # pattern's interfaces:
        assert self.ininterfaces(key[0])
        assert self.ininterfaces(key[1])

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
        return 'Pattern\n-------\n'+self.data.__repr__()

    def clear(self):
        """
        Clear all connections in class instance.
        """

        self.data.dropna(inplace=True)

    def isconnected(self, from_int, to_int):
        """
        Check whether the specified interfaces are connected.

        Parameters
        ----------
        from_int, to_int : int
            Interface identifiers; must be in `self.interface.keys()`.

        Returns
        -------
        result : bool
            True if at least one connection from `from_int` to `to_int`
            exists.
        """

        assert from_int != to_int
        assert from_int in self.interface
        assert to_int in self.interface

        # Get index of all defined connections:
        idx = self.data[self.data['conn'] != 0].index
        for t in idx.tolist():
            
            # Split tuple into 'from' and 'to' identifiers:
            from_id = t[0:self.num_levels['from']]
            to_id = t[self.num_levels['from']:self.num_levels['from']+self.num_levels['to']]
            if from_id in self.interface[from_int].index and \
               to_id in self.interface[to_int].index:
                return True
        return False

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

        # XXX this should refuse to load identifiers that are not in any of the
        # sets of ports comprised by the pattern:
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

        def test_create(self):
            c = Pattern('/foo[0:3]', '/bar[0:3]',
                        columns=['conn','from_type', 'to_type'])
            c['/foo[0]', '/bar[0]'] = [1, 'spike', 'spike']
            c['/foo[0]', '/bar[1]'] = [1, 'spike', 'spike']
            c['/foo[2]', '/bar[2]'] = [1, 'spike', 'spike']
            c['/bar[0]', '/foo[0]'] = [1, 'gpot', 'gpot']
            c['/bar[1]', '/foo[0]'] = [1, 'gpot', 'gpot']
            c['/bar[2]', '/foo[1]'] = [1, 'spike', 'gpot']
            assert_frame_equal(c.data, self.df)

    main()
