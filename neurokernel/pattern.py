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
    in_ports : list of tuple
        List of input port identifiers as tuples.
    out_ports : list of tuple
        List of output port identifiers as tuples.
    ports : list of tuple
        List of port identifiers as tuples.

    Parameters
    ----------
    selector : str
        Selector describing the port identifiers comprised by the interface.
    columns : list
        Data column names.

    Methods
    -------
    as_selectors(ids)
        Convert list of port identifiers to path-like selectors. 
    from_dict(d)
        Create an Interface from a dictionary of selectors and data values.
    
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

    @classmethod
    def from_dict(cls, d):
        """
        Create an Interface from a dictionary of selectors and data values.

        Parameters
        ----------
        d : dict
            Dictionary that maps selectors to the data that should be associated
            with the corresponding ports.
        
        Returns
        -------
        i : Interface
            Generated interface instance.
        """

        i = cls(','.join(d.keys()))
        for k, v in d.iteritems():
            i[k] = v
        i.data.sort_index(inplace=True)
        return i

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
        Add an additional level to the index of the pattern's internal DataFrame.
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
        if type(key) == tuple and len(key) > 1:
            return self.sel.select(self.data[list(key[1:])], key[0])
        else:
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

    @property
    def ports(self):
        """
        List of port identifiers as tuples.
        """

        return self.data.index.tolist()

    def is_compatible(self, other_int):
        """
        Check whether some Interface instance can be connected to the current instance.

        Parameters
        ----------
        other_int : Interface
            Interface instance to check.

        Returns
        -------
        result : bool
            True if both interfaces comprise the same identifiers
            and each identifier with an 'io' attribute set to 'out' in one
            interface has its 'io' attribute set to 'in' in the other interface.

        Notes
        -----
        All ports in both interfaces must have set 'io' attributes.
        """

        if not set(other_int.data['io']).issubset(['in', 'out']) or \
           not set(self.data['io']).issubset(['in', 'out']):
            raise ValueError("All ports must have their 'io' attribute set.")
        inv = self.data.applymap(lambda x: 'out' if x == 'in' else \
                                     ('in' if x == 'out' else x))
        if self.index.equals(other_int.index) and \
           all(inv['io'] == other_int.data['io']):
            return True
        else:
            return False

    @property
    def in_ports(self):
        """
        List of input port identifiers as list of tuples.
        """

        return self.data[self.data['io'] == 'in'].index.tolist()

    @property
    def out_ports(self):
        """
        List of output port identifiers as list of tuples.
        """

        return self.data[self.data['io'] == 'out'].index.tolist()

    @classmethod
    def as_selectors(cls, ids):
        """
        Convert list of port identifiers to path-like selectors.

        Parameters
        ----------
        ids : list of tuple
            Port identifiers.

        Returns
        -------
        selectors : list of str
            List of selector strings corresponding to each port identifier.
        """

        result = []
        for t in ids:
            selector = ''
            for s in t:
                if type(s) == str:
                    selector += '/'+s
                else:
                    selector += '[%s]' % s
            result.append(selector)
        return result

    def __len__(self):
        return self.data.__len__()

    def __repr__(self):
        return 'Interface\n---------\n'+self.data.__repr__()

class Pattern(object):
    """
    Connectivity pattern linking sets of interface ports.

    This class represents connection mappings between interfaces comprising sets
    of ports. Ports are represented using path-like identifiers [1]_; the
    presence of a row linking the two identifiers in the class' internal index
    indicates the presence of a connection. A single data attribute ('conn')
    associated with defined connections is created by default. Specific
    attributes may be accessed by specifying their names after the port
    identifiers; if a nonexistent attribute is specified when a sequential value
    is assigned, a new column for that attribute is automatically created: ::

        p['/x[0:3]', '/y[0:2]', 'conn', 'x'] = [1, 'foo']

    The direction of connections between ports in a class instance determines 
    whether they are input or output ports.

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
    interfaces : dict of Interface
        Interfaces containing port identifiers and attributes.

    Parameters
    ----------
    sel0, sel1, ...: str
        Selectors defining the sets of ports potentially connected by the 
        pattern. These selectors must be disjoint, i.e., no identifier 
        comprised by one selector may be in any other selector.
    columns : sequence of str
        Data column names.

    Methods
    -------
    clear()
        Clear all connections in class instance.
    dest_idx(src_int, dest_int, src_ports=None)
        Retrieve destination ports connected to the specified source ports.
    from_concat(*selectors, **kwargs)
        Create pattern from the concatenation of identifers comprised by two selectors.
    from_csv(file_name, **kwargs)
        Read connectivity data from CSV file.
    from_product(*selectors, **kwargs)
        Create pattern from the product of identifiers comprised by two selectors.
    ininterfaces(selector)
        Check whether a selector is supported by any of the pattern's interfaces.
    is_connected(from_int, to_int)
        Check whether the specified interfaces are connected.
    src_idx(src_int, dest_int, dest_ports=None)
        Retrieve source ports connected to the specified destination ports.
    whichint(selector)
        Return the interface containing the identifiers comprised by a selector.

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
        self.interfaces = {}
        max_levels = 0
        for i, s in enumerate(selectors):
            self.interfaces[i] = Interface(s)

            # Find the maximum number of levels required to accommodate all of 
            # the identifiers:
            if self.interfaces[i].num_levels > max_levels:
                max_levels = self.interfaces[i].num_levels

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

        # Update the `io` attributes of the pattern's interfaces:
        for t in p.sel.make_index(from_sel):
            p.interfaces[p.whichint([t])][[t], 'io'] = 'in'
        for t in p.sel.make_index(to_sel):
            p.interfaces[p.whichint([t])][[t], 'io'] = 'out'

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

    def whichint(self, selector):
        """
        Return the interface containing the identifiers comprised by a selector.
        """
        
        # If the selector is found in more than one interface, don't return any
        # of the interfaces:
        i = set()
        for k, v in self.interfaces.iteritems():
            if len(v[selector]):
                i.add(k)
        if len(i) == 1:
            return i.pop()
        else:
            return None

    def ininterfaces(self, selector):
        """
        Check whether a selector is supported by any of the pattern's interfaces.
        """

        for interface in self.interfaces.values():
            if len(interface[selector]) > 0:
                return True
        return False

    def __setitem__(self, key, value):
        # XXX attempting to create an index row that appears both in the 'from'
        # and 'to' sections of the pattern's index should raise an exception
        # because ports cannot both receive input and send output.

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

        # Update the `io` attributes of the pattern's interfaces:
        for t in self.sel.make_index(key[0]):
            self.interfaces[self.whichint([t])][[t], 'io'] = 'in'
        for t in self.sel.make_index(key[1]):
            self.interfaces[self.whichint([t])][[t], 'io'] = 'out'

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

        # If the specified selectors correspond to existing entries, set their attributes:
        if found:
            for k, v in data.iteritems():
                self.data[k].ix[idx] = v

        # Otherwise, populate a new DataFrame with the specified attributes:
        else:
            self.data = self.data.append(pd.DataFrame(data=data, index=idx))
            self.data.sort(inplace=True)

    def __getitem__(self, key):
        if len(key) > 2:
            return self.sel.select(self.data[list(key[2:])],
                                             selector = '+'.join(key[0:2]))
        else:
            return self.sel.select(self.data, selector = '+'.join(key))

    def src_idx(self, src_int, dest_int, dest_ports=None):
        """
        Retrieve source ports connected to the specified destination ports.

        Examples
        --------
        >>> p = Pattern('/foo[0:3]', '/bar[0:3]')
        >>> p['/foo[0:3]', '/bar[0:3]'] = 1
        >>> p['/bar[0:3]', '/foo[0:3]'] = 1
        >>> all(p.src_idx(0, 1, '/bar[0]') == [('foo', 0), ('foo', 1), ('foo', 2)])

        Parameters
        ----------
        src_int, dest_int : int
            Source and destination interface identifiers.
        dest_ports : str
            Path-like selector corresponding to ports in destination 
            interface. If not specified, all ports in the destination 
            interface are considered.

        Returns
        -------
        idx : list of tuple
            Source ports connected to the specified destination ports.
        """

        assert src_int != dest_int
        assert src_int in self.interfaces and dest_int in self.interfaces

        from_slice = slice(0, self.num_levels['from'])
        to_slice = slice(self.num_levels['from'],
                         self.num_levels['from']+self.num_levels['to'])
        if dest_ports is None:
            idx = self.data.select(lambda x: x[from_slice] in \
                        self.interfaces[src_int].index and \
                        x[to_slice] in \
                        self.interfaces[dest_int].index).index            
        else:
            idx = self.data.select(lambda x: x[from_slice] in \
                        self.interfaces[src_int].index and \
                        x[to_slice] in \
                        self.interfaces[dest_int][dest_ports].index).index
            
        # Don't include duplicate tuples in output:
        return list(set([x[from_slice] for x in idx]))

    def dest_idx(self, src_int, dest_int, src_ports=None):
        """
        Retrieve destination ports connected to the specified source ports.

        Examples
        --------
        >>> p = Pattern('/foo[0:3]', '/bar[0:3]')
        >>> p['/foo[0:3]', '/bar[0:3]'] = 1
        >>> p['/bar[0:3]', '/foo[0:3]'] = 1
        >>> all(p.dest_idx(0, 1, '/foo[0]') == [('bar', 0), ('bar', 1), ('bar', 2)])

        Parameters
        ----------
        src_int, dest_int : int
            Source and destination interface identifiers.
        src_ports : str
            Path-like selector corresponding to ports in source
            interface. If not specified, all ports in the source
            interface are considered.

        Returns
        -------
        idx : list of tuple
            Destination ports connected to the specified source ports.
        """

        assert src_int != dest_int
        assert src_int in self.interfaces and dest_int in self.interfaces

        from_slice = slice(0, self.num_levels['from'])
        to_slice = slice(self.num_levels['from'],
                         self.num_levels['from']+self.num_levels['to'])
        if src_ports is None:
            idx = self.data.select(lambda x: x[from_slice] in \
                        self.interfaces[src_int].index and \
                        x[to_slice] in \
                        self.interfaces[dest_int].index).index
        else:
            idx = self.data.select(lambda x: x[from_slice] in \
                        self.interfaces[src_int][src_ports].index and \
                        x[to_slice] in \
                        self.interfaces[dest_int].index).index

        # Don't include duplicate tuples in output:
        return list(set([x[to_slice] for x in idx]))

    def __len__(self):
        return self.data.__len__()

    def __repr__(self):
        return 'Pattern\n-------\n'+self.data.__repr__()

    def clear(self):
        """
        Clear all connections in class instance.
        """

        self.data.drop(self.data.index, inplace=True)

    def is_connected(self, from_int, to_int):
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
        assert from_int in self.interfaces
        assert to_int in self.interfaces

        # Get index of all defined connections:
        idx = self.data[self.data['conn'] != 0].index
        for t in idx.tolist():
            
            # Split tuple into 'from' and 'to' identifiers:
            from_id = t[0:self.num_levels['from']]
            to_id = t[self.num_levels['from']:self.num_levels['from']+self.num_levels['to']]
            if from_id in self.interfaces[from_int].index and \
               to_id in self.interfaces[to_int].index:
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
    from pandas.util.testing import assert_frame_equal, assert_index_equal

    class test_interface(TestCase):
        def setUp(self):
            self.interface = Interface('/foo[0:3]')
            self.interface['/foo[0]', 'io'] = 'in'
            self.interface['/foo[1:3]', 'io'] = 'out'

        def test_index(self):
            assert_index_equal(self.interface.index,
                               pd.MultiIndex(levels=[['foo'], [0, 1, 2]],
                                             labels=[[0, 0, 0], [0, 1, 2]],
                                             names=['0', '1']))

        def test_ports(self):
            self.assertSequenceEqual(self.interface.ports,
                                     [('foo', 0),
                                      ('foo', 1),
                                      ('foo', 2)])

        def test_in_ports(self):
            self.assertSequenceEqual(self.interface.in_ports,
                                     [('foo', 0)])

        def test_out_ports(self):
            self.assertSequenceEqual(self.interface.out_ports,
                                     [('foo', 1), ('foo', 2)])

        def test_as_selectors(self):
            self.assertSequenceEqual(self.interface.as_selectors([('foo', 0),
                                                                  ('foo', 1)]),
                                     ['/foo[0]', '/foo[1]'])

        def test_is_compatible_both_dirs(self):
            i = Interface('/foo[0:3]')
            i['/foo[0]', 'io'] = 'out'
            i['/foo[1:3]', 'io'] = 'in'
            j = Interface('/foo[0:3]')
            j['/foo[0]', 'io'] = 'in'
            j['/foo[1:3]', 'io'] = 'out'
            assert i.is_compatible(j)

        def test_is_compatible_one_dir(self):
            i = Interface('/foo[0:3]')
            i['/foo[0:3]', 'io'] = 'out'
            j = Interface('/foo[0:3]')
            j['/foo[0:3]', 'io'] = 'in'
            assert i.is_compatible(j)

    class test_pattern(TestCase):
        def setUp(self):
            # XXX not a good example; a pattern shouldn't allow a single port to
            # both send output and receive input:
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
            p = Pattern('/foo[0:3]', '/bar[0:3]',
                        columns=['conn','from_type', 'to_type'])
            p['/foo[0]', '/bar[0]'] = [1, 'spike', 'spike']
            p['/foo[0]', '/bar[1]'] = [1, 'spike', 'spike']
            p['/foo[2]', '/bar[2]'] = [1, 'spike', 'spike']
            p['/bar[0]', '/foo[0]'] = [1, 'gpot', 'gpot']
            p['/bar[1]', '/foo[0]'] = [1, 'gpot', 'gpot']
            p['/bar[2]', '/foo[1]'] = [1, 'spike', 'gpot']
            assert_frame_equal(p.data, self.df)

        def test_src_idx_all(self):
            p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
            p['/aaa[0:3]', '/yyy[0:3]'] = 1
            p['/xxx[0:3]', '/bbb[0:3]'] = 1
            self.assertItemsEqual(p.src_idx(0, 1),
                                  [('aaa', 0),
                                   ('aaa', 1),
                                   ('aaa', 2)])
            
        def test_src_idx_specific(self):
            p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
            p['/aaa[0]', '/yyy[0]'] = 1
            p['/aaa[1:3]', '/yyy[1:3]'] = 1
            p['/xxx[0:3]', '/bbb[0:3]'] = 1
            self.assertItemsEqual(p.src_idx(0, 1, '/yyy[0]'),
                                  [('aaa', 0)])
            
        def test_dest_idx_all(self):
            p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
            p['/aaa[0:3]', '/yyy[0:3]'] = 1
            p['/xxx[0:3]', '/bbb[0:3]'] = 1
            self.assertItemsEqual(p.dest_idx(0, 1, '/aaa[0]'),
                                  [('yyy', 0),
                                   ('yyy', 1),
                                   ('yyy', 2)])

        def test_dest_idx_specific(self):
            p = Pattern('/[aaa,bbb][0:3]', '/[xxx,yyy][0:3]')
            p['/aaa[0]', '/yyy[0]'] = 1
            p['/aaa[1:3]', '/yyy[1:3]'] = 1
            p['/xxx[0:3]', '/bbb[0:3]'] = 1
            self.assertItemsEqual(p.dest_idx(0, 1, '/aaa[0]'),
                                  [('yyy', 0)])

        def test_is_connected(self):
            p = Pattern('/aaa[0:3]', '/bbb[0:3]')
            p['/aaa[0]', '/bbb[2]'] = 1
            assert p.is_connected(0, 1) == True
            assert p.is_connected(1, 0) == False

        def test_clear(self):
            p = Pattern('/aaa[0:3]', '/bbb[0:3]')
            p['/aaa[0:3]', '/bbb[0:3]'] = 1
            p.clear()
            assert len(p) == 0

    main()
