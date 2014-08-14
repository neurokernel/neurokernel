#!/usr/bin/env python

"""
Represent connectivity pattern using pandas DataFrame.
"""

from collections import OrderedDict
import itertools

from chash import lfu_cache_method
import networkx as nx
import numpy as np
import pandas as pd

from plsel import PathLikeSelector

from chash import chash
_hash_func = chash

class Interface(object):
    """
    Container for set of interface comprising ports.

    This class contains information about a set of interfaces comprising
    path-like identifiers and the attributes associated with them.
    By default, each port must have at least the following attributes;
    other attributes may be added:

    - interface - indicates which interface a port is associated with.
    - io - indicates whether the port receives input ('in') or
      emits output ('out').
    - type - indicates whether the port emits/receives spikes or
      graded potentials.

    All port identifiers in an interface must be unique. For two interfaces
    to be deemed compatible, they must contain the same port identifiers and
    their identifiers' 'io' attributes must be the inverse of each other
    (i.e., every 'in' port in one interface must be mirrored by an 'out' port
    in the other interface.

    Examples
    --------
    >>> i = Interface('/foo[0:4],/bar[0:3]')
    >>> i['/foo[0:2]', 'interface', 'io', 'type'] = [0, 'in', 'spike']
    >>> i['/foo[2:4]', 'interface', 'io', 'type'] = [1, 'out', 'spike']

    Attributes
    ----------
    data : pandas.DataFrame
        Port attribute data.
    index : pandas.MultiIndex
        Index of port identifiers.

    Parameters
    ----------
    selector : str, unicode, or sequence
            Selector string (e.g., 'foo[0:2]') or sequence of token
            sequences (e.g., [['foo', (0, 2)]]) describing the port
            identifiers comprised by the interface.
    columns : list, default = ['interface', 'io', 'type']
        Data column names.

    See Also
    --------
    plsel.PathLikeSelector
    """

    def __init__(self, selector='', columns=['interface', 'io', 'type']):

        # All ports in an interface must contain at least the following
        # attributes:
        assert set(columns).issuperset(['interface', 'io', 'type'])
        self.sel = PathLikeSelector()
        assert not(self.sel.is_ambiguous(selector))
        self.num_levels = self.sel.max_levels(selector)
        names = [str(i) for i in xrange(self.num_levels)]
        idx = self.sel.make_index(selector, names)
        self.__validate_index__(idx)
        self.data = pd.DataFrame(index=idx, columns=columns, dtype=object)

    def __validate_index__(self, idx):
        """
        Raise an exception if the specified index will result in an invalid interface.
        """

        if (hasattr(idx, 'has_duplicates') and idx.has_duplicates) or \
                len(idx.unique()) < len(idx):
            raise ValueError('Duplicate interface index entries detected.')

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

        # Try using the selector to select data from the internal DataFrame:
        try:
            idx = self.sel.get_index(self.data, selector,
                                     names=self.data.index.names)

        # If the select fails, try to create new rows with the index specified
        # by the selector and load them with the specified data:
        except ValueError:
            try:
                idx = self.sel.make_index(selector, self.data.index.names)
            except:
                raise ValueError('cannot create index with '
                                 'selector %s and column names %s' \
                                 % (selector, str(self.data.index.names)))
            else:
                found = False
        else:
            found = True

        # If the data specified is not a dict, convert it to a dict:
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
            new_data = self.data.append(pd.DataFrame(data=data, index=idx, 
                                                     dtype=object))

            # Validate updated DataFrame's index before updating the instance's
            # data attribute:
            self.__validate_index__(new_data.index)
            self.data = new_data
            self.data.sort(inplace=True)

    @property
    def index(self):
        """
        Interface index.
        """

        return self.data.index

    @index.setter
    def index(self, i):
        self.data.index = i

    @property
    def interface_ids(self):
        """
        Interface identifiers.
        """

        return set(self.data['interface'])

    @property
    def io_inv(self):
        """
        Returns new Interface instance with inverse input-output attributes.

        Returns
        -------
        i : Interface
            Interface instance whose 'io' attributes are the inverse of those of
        the current instance.
        """

        data_inv = self.data.copy()
        f = lambda x: 'out' if x == 'in' else \
            ('in' if x == 'out' else x)
        data_inv['io'] = data_inv['io'].apply(f)
        return self.from_df(data_inv)

    def clear(self):
        """
        Clear all ports in class instance.
        """

        self.data.drop(self.data.index, inplace=True)

    def data_select(self, f, inplace=False):
        """
        Restrict Interface data with a selection function.

        Returns an Interface instance containing only those rows
        whose data is passed by the specified selection function.

        Parameters
        ----------
        f : function
            Selection function with a single dict argument whose keys
            are the Interface's data column names.
        inplace : bool, default=False
            If True, update and return the given Interface instance.
            Otherwise, return a new instance.

        Returns
        -------
        i : Interface
            Interface instance containing data selected by `f`.
        """

        assert callable(f)
        result = self.data[f(self.data)]
        if inplace:
            self.data = result
            return self
        else:
            return Interface.from_df(result)

    @classmethod
    def from_df(cls, df):
        """
        Create an Interface from a properly formatted DataFrame.

        Examples
        --------
        >>> import plsel
        >>> import pandas
        >>> idx = plsel.make_index('/foo[0:2]')
        >>> data = [[0, 'in', 'spike'], [1, 'out', 'gpot']]
        >>> columns = ['interface', 'io', 'type']
        >>> df = pandas.DataFrame(data, index=idx, columns=columns)

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with a MultiIndex and data columns 'interface', 
            'io', and 'type' (additional columns may also be present).

        Returns
        -------
        i : Interface
            Generated Interface instance.

        Notes
        -----
        The contents of the specified DataFrame instance are copied into the
        new Interface instance.
        """
        
        assert set(df.columns).issuperset(['interface', 'io', 'type'])
        if isinstance(df.index, pd.MultiIndex):
            i = cls(df.index.tolist(), df.columns)
        elif isinstance(df.index, pd.Index):
            i = cls([(s,) for s in df.index.tolist()], df.columns)
        else:
            raise ValueError('invalid index type')
        i.data = df.copy()
        i.__validate_index__(i.index)
        return i

    @classmethod
    def from_dict(cls, d):
        """
        Create an Interface from a dictionary of selectors and data values.

        Examples
        --------
        >>> d = {'/foo[0]': [0, 'in', 'gpot'], '/foo[1]': [1, 'in', 'gpot']}
        >>> i = Interface.from_dict(d)
        
        Parameters
        ----------
        d : dict
            Dictionary that maps selectors to the data that should be associated
            with the corresponding ports. If a scalar, the data is assigned to
            the first attribute; if an iterable, the data is assigned to the
            attributes in order.
        
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

    @classmethod
    def from_graph(cls, g):
        """
        Create an Interface from a NetworkX graph.

        Examples
        --------
        >>> import networkx as nx
        >>> g = nx.Graph()
        >>> g.add_node('/foo[0]', interface=0, io='in', type='gpot')
        >>> g.add_node('/foo[1]', interface=0, io='in', type='gpot')
        >>> i = Interface.from_graph(g)
        
        Parameters
        ----------
        g : networkx.Graph
            Graph whose node IDs are path-like port identifiers. The node attributes
            are assigned to the ports.
        
        Returns
        -------
        i : Interface
            Generated interface instance.
        """

        assert isinstance(g, nx.Graph)
        return cls.from_dict(g.node)

    def gpot_ports(self, i=None):
        """
        Restrict Interface ports to graded potential ports.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return Interface instance containing
            all graded potential ports.

        Returns
        -------
        interface : Interface
            Interface instance containing all graded potential ports and their attributes
            in the specified interface.
        """

        if i is None:
            try:
                return self.from_df(self.data[self.data['type'] == 'gpot'])
            except:
                return Interface()
        else:
            try:
                return self.from_df(self.data[(self.data['type'] == 'gpot') & \
                                              (self.data['interface'] == i)])
            except:
                return Interface()

    def in_ports(self, i=None):
        """
        Restrict Interface ports to input ports.

        Parameters
        ----------
        i : int
            Interface identifier.

        Returns
        -------
        interface : Interface
            Interface instance containing all input ports and their attributes
            in the specified interface.
        """

        if i is None:
            try:
                return self.from_df(self.data[self.data['io'] == 'in'])
            except:
                return Interface()
        else:
            try:
                return self.from_df(self.data[(self.data['io'] == 'in') & \
                                              (self.data['interface'] == i)])
            except:
                return Interface()

    def interface_ports(self, i=None):
        """
        Restrict Interface ports to specific interface.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return Interface instance containing
            all ports.

        Returns
        -------
        interface : Interface
            Interface instance containing all ports and attributes in the
            specified interface.
        """

        if i is None:
            return self.copy()
        else:
            try:
                return self.from_df(self.data[self.data['interface'] == i])
            except:
                return Interface()

    def is_compatible(self, a, i, b):
        """
        Check whether two interfaces can be connected.

        Compares an interface in the current Interface instance with one in
        another instance to determine whether their ports can be connected.

        Parameters
        ----------
        a : int
            Identifier of interface in the current instance.
        i : Interface
            Interface instance containing the other interface.
        b : int
            Identifier of interface in instance `i`.

        Returns
        -------
        result : bool
            True if both interfaces comprise the same identifiers, the set 'type'
            attributes for each matching pair of identifiers in the two
            interfaces match, and each identifier with an 'io' attribute set 
            to 'out' in one interface has its 'io' attribute set to 'in' in the 
            other interface.

        Notes
        -----
        Assumes that the port identifiers in both interfaces are sorted in the
        same order.
        """

        assert isinstance(i, Interface)

        # Find 'type' attributes for specified interfaces:
        type_a = self.data[self.data['interface'] == a]['type']
        type_b = i.data[i.data['interface'] == b]['type']

        # Exclude null entries from 'type' attribs:
        type_a = type_a[type_a.notnull()]
        type_b = type_b[type_b.notnull()]

        # Find inverse of this instance's 'io' attributes 
        # for interface 'a' and 'io' attributes for interface 'b':
        f = lambda x: 'out' if x == 'in' else \
            ('in' if x == 'out' else x)
        io_a_inv = self.data[self.data['interface'] == a]['io'].apply(f)
        io_b = i.data[i.data['interface'] == b]['io']
        # Exclude null entries from inverted and original 'io' attribs:
        io_a_inv = io_a_inv[io_a_inv.notnull()]
        io_b = io_b[io_b.notnull()]

        # Compare indices, non-null 'io' attribs, and non-null 'type' attribs:
        idx_a = self.data[self.data['interface'] == a].index
        idx_b = i.data[i.data['interface'] == b].index
        if idx_a.equals(idx_b) and all(io_a_inv==io_b) \
                and all(type_a==type_b):
            return True
        else:
            return False

    def is_in_interfaces(self, s):
        """
        Check whether ports comprised by a selector are in the stored interfaces.
        
        Parameters
        ----------
        s : str or unicode
            Port selector.

        Returns
        -------
        result : bool
            True if the comprised ports are in any of the stored interfaces.
        """

        return self.sel.is_in(s, self.index.tolist())

    def out_ports(self, i=None):
        """
        Restrict Interface ports to output ports.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return Interface instance containing
            all output ports.

        Returns
        -------
        interface : Interface
            Interface instance containing all output ports and their attributes
            in the specified interface.
        """

        if i is None:
            try:
                return self.from_df(self.data[self.data['io'] == 'out'])
            except:
                return Interface()
        else:
            try:
                return self.from_df(self.data[(self.data['io'] == 'out') & \
                                              (self.data['interface'] == i)])
            except:
                return Interface()

    def port_select(self, f, inplace=False):
        """
        Restrict Interface ports with a selection function.

        Returns an Interface instance containing only those rows
        whose ports are passed by the specified selection function.

        Parameters
        ----------
        f : function
            Selection function with a single tuple argument containing
            the various columns of the Interface instance's MultiIndex.
        inplace : bool, default=False
            If True, update and return the given Interface instance.
            Otherwise, return a new instance.

        Returns
        -------
        i : Interface
            Interface instance containing ports selected by `f`.
        """

        assert callable(f)
        if inplace:
            self.data = self.data.select(f)
            return self
        else:
            return Interface.from_df(self.data.select(f))

    def spike_ports(self, i=None):
        """
        Restrict Interface ports to spiking ports.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return Interface instance containing
            all spiking ports.

        Returns
        -------
        interface : Interface
            Interface instance containing all spiking ports and their attributes
            in the specified interface.
        """

        if i is None:
            try:
                return self.from_df(self.data[self.data['type'] == 'spike'])
            except:
                return Interface()
        else:
            try:
                return self.from_df(self.data[(self.data['type'] == 'spike') & \
                                              (self.data['interface'] == i)])
            except:
                return Interface()

    def to_selectors(self, i=None):
        """
        Retrieve Interface's port identifiers as list of path-like selectors.

        Parameters
        ----------
        i : int
            Interface identifier. If set to None, return all port identifiers.

        Returns
        -------
        selectors : list of str
            List of selector strings corresponding to each port identifier.
        """

        ids = self.to_tuples(i)
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

    def to_tuples(self, i=None):
        """
        Retrieve Interface's port identifiers as list of tuples.

        Parameters
        ----------
        i : int
            Interface identifier. If set to None, return all port identifiers.

        Returns
        -------
        result : list of tuple
            List of token tuples corresponding to each port identifier.
        """

        if i is None:
            if isinstance(self.index, pd.MultiIndex):
                return self.index.tolist()
            else:
                return [(t,) for t in self.index]
        try:
            if isinstance(self.index, pd.MultiIndex):
                return self.data[self.data['interface'] == i].index.tolist()
            else:
                return [(t,) for t in self.data[self.data['interface'] == i].index]
        except:
            return []
    
    def which_int(self, s):
        """
        Return the interface containing the identifiers comprised by a selector.

        Parameters
        ----------
        selector : str or unicode
            Port selector.

        Returns
        -------
        i : set
            Set of identifiers for interfaces that contain ports comprised by
            the selector.
        """
        
        try:
            s = set(self[s, 'interface'].values.flatten())

            # Ignore unset entries:
            s.discard(np.nan)
            return s
        except KeyError:
            return set()

    def __copy__(self):
        """
        Make a copy of this object.
        """

        return self.from_df(self.data)
    copy = __copy__
    copy.__doc__ = __copy__.__doc__

    def __len__(self):
        return self.data.__len__()

    def __repr__(self):
        return 'Interface\n---------\n'+self.data.__repr__()

class Pattern(object):
    """
    Connectivity pattern linking sets of interface ports.

    This class represents connection mappings between interfaces comprising 
    sets of ports. Ports are represented using path-like identifiers; 
    the presence of a row linking the two identifiers in the class' internal 
    index indicates the presence of a connection. A single data attribute 
    ('conn') associated with defined connections is created by default. 
    Specific attributes may be accessed by specifying their names after the 
    port identifiers; if a nonexistent attribute is specified when a sequential 
    value is assigned, a new column for that attribute is automatically 
    created: ::

        p['/x[0]', '/y[0]', 'conn', 'x'] = [1, 'foo']

    The direction of connections between ports in a class instance determines 
    whether they are input or output ports. Ports may not both receive input or 
    emit output. Patterns may contain fan-out connections, i.e., one source port
    connected to multiple destination ports, but not fan-in connections, i.e.,
    multiple source ports connected to a single destination port.

    Examples
    --------
    >>> p = Pattern('/x[0:3]','/y[0:4]')
    >>> p['/x[0]', '/y[0:2]'] = 1
    >>> p['/y[2]', '/x[1]'] = 1
    >>> p['/y[3]', '/x[2]'] = 1

    Attributes
    ----------
    data : pandas.DataFrame
        Connection attribute data.
    index : pandas.MultiIndex
        Index of connections.
    interface : Interface
        Interfaces containing port identifiers and attributes.

    Parameters
    ----------
    sel0, sel1, ...: str, unicode, or sequence
        Selectors defining the sets of ports potentially connected by the 
        pattern. These selectors must be disjoint, i.e., no identifier 
        comprised by one selector may be in any other selector.
    columns : sequence of str
        Data column names.

    See Also
    --------
    plsel.PathLikeSelector
    """

    def __init__(self, *selectors, **kwargs):
        columns = kwargs['columns'] if kwargs.has_key('columns') else ['conn']
        self.sel = PathLikeSelector()

        # Force sets of identifiers to be disjoint so that no identifier can
        # denote a port in more than one set:
        assert self.sel.are_disjoint(*selectors)

        # Collect all of the selectors:
        selector = []
        for s in selectors:
            if type(s) in [str, unicode]:
                selector.extend(self.sel.parse(s))
            elif np.iterable(s):
                selector.extend(s)
            else:
                raise ValueError('invalid selector type')

        # Create Interface instance containing the ports comprised by all of the
        # specified selectors:
        self.interface = Interface(selector)

        # Set the interface identifiers associated with each of the selectors
        # consecutively:
        for i, s in enumerate(selectors):
            self.interface[s, 'interface'] = i

        # Create a MultiIndex that can store mappings between identifiers in the
        # two interfaces:
        self.num_levels = {'from': self.interface.num_levels,
                           'to': self.interface.num_levels}
        names = ['from_%s' % i for i in xrange(self.num_levels['from'])]+ \
                ['to_%s' %i for i in xrange(self.num_levels['to'])]
        levels = [[] for i in xrange(len(names))]
        labels = [[] for i in xrange(len(names))]
        idx = pd.MultiIndex(levels=levels, labels=labels, names=names)
                            
        self.data = pd.DataFrame(index=idx, columns=columns, dtype=object)

    @property
    def from_slice(self):
        """
        Slice of pattern index row corresponding to source port(s).
        """

        return slice(0, self.num_levels['from'])

    @property
    def to_slice(self):
        """
        Slice of pattern index row corresponding to destination port(s).
        """

        return slice(self.num_levels['from'],        
                     self.num_levels['from']+self.num_levels['to'])

    @property
    def index(self):
        """
        Pattern index.
        """

        return self.data.index
    @index.setter
    def index(self, i):
        self.data.index = i

    @property
    def interface_ids(self):
        """
        Interface identifiers.
        """

        return self.interface.interface_ids

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
        p.__validate_index__(idx)

        # Replace the pattern's DataFrame:
        p.data = pd.DataFrame(data=data, index=idx, columns=columns)

        # Update the `io` attributes of the pattern's interfaces:
        p.interface[from_sel, 'io'] = 'in'
        p.interface[to_sel, 'io'] = 'out'

        return p

    def clear(self):
        """
        Clear all connections in class instance.
        """

        self.interface.clear()
        self.data.drop(self.data.index, inplace=True)

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

    def gpot_ports(self, i=None):
        return self.interface.gpot_ports(i)
    gpot_ports.__doc__ = Interface.gpot_ports.__doc__

    def in_ports(self, i=None):
        return self.interface.in_ports(i)
    in_ports.__doc__ = Interface.in_ports.__doc__

    def interface_ports(self, i=None):
        return self.interface.interface_ports(i)
    interface_ports.__doc__ = Interface.interface_ports.__doc__

    def out_ports(self, i=None):
        return self.interface.out_ports(i)
    out_ports.__doc__ = Interface.out_ports.__doc__

    def spike_ports(self, i=None):
        return self.interface.spike_ports(i)
    spike_ports.__doc__ = Interface.spike_ports.__doc__

    @classmethod
    def from_concat(cls, *selectors, **kwargs):
        """
        Create pattern from the concatenation of identifers in two selectors.

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

    def __validate_index__(self, idx):
        """
        Raise an exception if the specified index will result in an invalid pattern.
        """

        # Prohibit duplicate connections:
        if (hasattr(idx, 'has_duplicates') and idx.has_duplicates) or \
           len(idx.unique()) < len(idx):
            raise ValueError('Duplicate pattern entries detected.')
            
        # Prohibit fan-in connections (i.e., patterns whose index has duplicate
        # 'from' port identifiers):
        from_idx, to_idx = self.split_multiindex(idx, 
                                                 self.from_slice, self.to_slice)
        if (hasattr(to_idx, 'has_duplicates') and to_idx.has_duplicates) or \
           len(to_idx.unique()) < len(to_idx):
            raise ValueError('Fan-in pattern entries detected.')

        # Prohibit ports that both receive input and send output:
        if set(from_idx).intersection(to_idx):
            raise ValueError('Ports cannot both receive input and send output.')

    def which_int(self, s):
        return self.interface.which_int(s)
    which_int.__doc__ = Interface.which_int.__doc__

    def is_in_interfaces(self, selector):
        """
        Check whether a selector is supported by any stored interface.
        """

        if len(self.interface[selector]) > 0:
            return True
        else:
            return False

    def get_conns(self, as_str=False):
        """
        Return connections as pairs of port identifiers.
        
        Parameters
        ----------
        as_str : bool
            If True, return connections as a list of identifier
            string pairs. Otherwise, return them as pairs of token tuples.
        """

        if as_str:
            return [(self.sel.to_identifier(row[self.from_slice]),
                     self.sel.to_identifier(row[self.to_slice])) \
                    for row in self.data.index]
        else:
            return [(row[self.from_slice], row[self.to_slice]) \
                    for row in self.data.index]

    def __setitem__(self, key, value):
        # Must pass more than one argument to the [] operators:
        assert type(key) == tuple

        # Ensure that specified selectors refer to ports in the
        # pattern's interfaces:
        assert self.is_in_interfaces(key[0])
        assert self.is_in_interfaces(key[1])
        
        # Ensure that the ports are in different interfaces:
        assert self.which_int(key[0]) != self.which_int(key[1])

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

        # If the specified selectors correspond to existing entries, 
        # set their attributes:
        if found:
            for k, v in data.iteritems():
                self.data[k].ix[idx] = v

        # Otherwise, populate a new DataFrame with the specified attributes:
        else:
            new_data = self.data.append(pd.DataFrame(data=data, index=idx,
                                                     dtype=object))

            # Validate updated DataFrame's index before updating the instance's
            # data attribute:
            self.__validate_index__(new_data.index)
            self.data = new_data
            self.data.sort(inplace=True)

        # Update the `io` attributes of the pattern's interfaces:
        self.interface[key[0], 'io'] = 'in'
        self.interface[key[1], 'io'] = 'out'

    def __getitem__(self, key):
        if len(key) > 2:
            return self.sel.select(self.data[list(key[2:])],
                                             selector = '+'.join(key[0:2]))
        else:
            return self.sel.select(self.data, selector = '+'.join(key))

    def src_idx(self, src_int, dest_int, 
                src_type=None, dest_type=None, dest_ports=None):                
        """
        Retrieve source ports connected to the specified destination ports.

        Examples
        --------
        >>> p = Pattern('/foo[0:4]', '/bar[0:4]')
        >>> p['/foo[0]', '/bar[0]'] = 1
        >>> p['/foo[1]', '/bar[1]'] = 1
        >>> p['/foo[2]', '/bar[2]'] = 1
        >>> p['/bar[3]', '/foo[3]'] = 1
        >>> all(p.src_idx(0, 1, dest_ports='/bar[0,1]') == [('foo', 0), ('foo', 1)])
        True

        Parameters
        ----------
        src_int, dest_int : int
            Source and destination interface identifiers.
        src_type, dest_type : str
            Types of source and destination ports as listed in their respective 
            interfaces.
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
        assert src_int in self.interface.interface_ids and \
            dest_int in self.interface.interface_ids

        # Filter destination ports by specified type:
        if dest_type is None:
            to_int = self.interface.interface_ports(dest_int)
        else:
            to_f = lambda x: x['type'] == dest_type
            to_int = self.interface.interface_ports(dest_int).data_select(to_f)

        # Filter destination ports by specified ports:
        if dest_ports is None:
            to_idx = to_int.index
        else:
            to_idx = to_int[dest_ports].index

        # Filter source ports by specified type:
        if src_type is None:
            from_int = self.interface.interface_ports(src_int)
        else:
            from_f = lambda x: x['type'] == src_type
            from_int = self.interface.interface_ports(src_int).data_select(from_f)

        from_idx = from_int.index

        # Construct index from those rows in the pattern whose ports have been
        # selected by the above code:
        idx = self.data.select(lambda x: x[self.from_slice] in from_idx \
                               and x[self.to_slice] in to_idx).index
                
        # Remove duplicate tuples from output without perturbing the order
        # of the remaining tuples:
        return OrderedDict.fromkeys([x[self.from_slice] for x in idx]).keys()

    def dest_idx(self, src_int, dest_int, 
                 src_type=None, dest_type=None, src_ports=None):
        """
        Retrieve destination ports connected to the specified source ports.

        Examples
        --------
        >>> p = Pattern('/foo[0:4]', '/bar[0:4]')
        >>> p['/foo[0]', '/bar[0]'] = 1
        >>> p['/foo[1]', '/bar[1]'] = 1
        >>> p['/foo[2]', '/bar[2]'] = 1
        >>> p['/bar[3]', '/foo[3]'] = 1
        >>> all(p.dest_idx(0, 1, src_ports='/foo[0,1]') == [('bar', 0), ('bar', 1)])
        True

        Parameters
        ----------
        src_int, dest_int : int
            Source and destination interface identifiers.
        src_type, dest_type : str
            Types of source and destination ports as listed in their respective 
            interfaces.
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
        assert src_int in self.interface.interface_ids and \
            dest_int in self.interface.interface_ids

        # Filter source ports by specified type:
        if src_type is None:
            from_int = self.interface.interface_ports(src_int)
        else:
            from_f = lambda x: x['type'] == src_type
            from_int = self.interface.interface_ports(src_int).data_select(from_f)

        # Filter source ports by specified ports:
        if src_ports is None:
            from_idx = from_int.index    
        else:
            from_idx = from_int[src_ports].index
            
        # Filter destination ports by specified type:
        if dest_type is None:
            to_int = self.interface.interface_ports(dest_int)
        else:
            to_f = lambda x: x['type'] == dest_type
            to_int = self.interface.interface_ports(dest_int).data_select(to_f)

        to_idx = to_int.index

        # Construct index from those rows in the pattern whose ports have been
        # selected by the above code:
        idx = self.data.select(lambda x: x[self.from_slice] in from_idx \
                               and x[self.to_slice] in to_idx).index

        # Remove duplicate tuples from output without perturbing the order
        # of the remaining tuples:
        return OrderedDict.fromkeys([x[self.to_slice] for x in idx]).keys()

    def __len__(self):
        return self.data.__len__()

    def __repr__(self):
        return 'Pattern\n-------\n'+self.data.__repr__()

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
        assert from_int in self.interface.interface_ids
        assert to_int in self.interface.interface_ids
        
        # Get index of all defined connections:
        idx = self.data[self.data['conn'] != 0].index
        from_ind_list = self.interface.interface_ports(from_int).index
        to_ind_list = self.interface.interface_ports(to_int).index
        for t in idx.tolist():
            
            # Split tuple into 'from' and 'to' identifiers; since the interface
            # index for a 'from' or 'to' identifier is an Index rather than a
            # MultiIndex, we need to extract a scalar rather than a tuple in the
            # former case:
            if self.num_levels['from'] == 1:
                from_id = t[0]
            else:
                from_id = t[0:self.num_levels['from']]
            if self.num_levels['to'] == 1:
                to_id = t[self.num_levels['from']]
            else:
                to_id = t[self.num_levels['from']:self.num_levels['from']+self.num_levels['to']]

            if from_id in from_ind_list and \
               to_id in to_ind_list:
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

    @classmethod
    def from_graph(cls, g):
        """
        Convert a NetworkX directed graph into a Pattern instance.

        Parameters
        ----------
        g : networkx.DiGraph
            Graph to convert.

        Returns
        -------
        p : Pattern
            Pattern instance.

        Notes
        -----
        The nodes in the specified graph must contain an 'interface' attribute.
        """

        assert type(g) == nx.DiGraph

        # Group ports by interface number:
        ports_by_int = {}
        for n, data in g.nodes(data=True):
            assert PathLikeSelector.is_identifier(n)
            assert data.has_key('interface')
            if not ports_by_int.has_key(data['interface']):
                ports_by_int[data['interface']] = {}
            ports_by_int[data['interface']][n] = data

        # Create selectors for each interface number:
        selector_list = []
        for interface in sorted(ports_by_int.keys()):
            selector_list.append(','.join(ports_by_int[interface].keys()))

        p = cls(*selector_list)
        for n, data in g.nodes(data=True):
            p.interface[n] = data
        for c in g.edges():
            p[c[0], c[1]] = 1

        p.data.sort_index(inplace=True)
        p.interface.data.sort_index(inplace=True)
        return p

    @classmethod
    def split_multiindex(cls, idx, a, b):
        """
        Split a single MultiIndex into two instances.

        Parameters
        ----------
        idx : pandas.MultiIndex
            MultiIndex to split.
        a, b : slice
            Ranges of index columns to include in the two resulting instances.

        Returns
        -------
        idx_a, idx_b : pandas.MultiIndex
            Resulting MultiIndex instances.
        """

        t_list = idx.tolist()
        idx_a = pd.MultiIndex.from_tuples([t[a] for t in t_list])
        idx_b = pd.MultiIndex.from_tuples([t[b] for t in t_list])
        return idx_a, idx_b

    def to_graph(self):
        """
        Convert the pattern to a networkx directed graph.
        
        Returns
        -------
        g : networkx.DiGraph
            Graph whose nodes are the pattern's ports 
            and whose edges are the pattern's connections.

        Notes
        -----
        The 'conn' attribute of the connections is not transferred to the graph
        edges.

        This method relies upon the assumption that the sets of 
        port identifiers comprised by the pattern's interfaces are disjoint.
        """

        g = nx.DiGraph()

        # Add all of the ports as nodes:
        for t in self.interface.data.index:    
            id = self.sel.to_identifier(t)

            # Replace NaNs with empty strings:
            d = {k: (v if str(v) != 'nan' else '') \
                 for k, v in self.interface.data.ix[t].to_dict().iteritems()}

            # Each node's name corresponds to the port identifier string:
            g.add_node(id, d)

        # Add all of the connections as edges:
        for t in self.data.index:
            t_from = t[self.from_slice]
            t_to = t[self.to_slice]
            id_from = self.sel.to_identifier(t_from)
            id_to = self.sel.to_identifier(t_to)
            d = self.data.ix[t].to_dict()

            # Discard the 'conn' attribute because the existence of the edge
            # indicates that the connection exists:
            if d.has_key('conn'):
                d.pop('conn')

            g.add_edge(id_from, id_to, d)

        return g
