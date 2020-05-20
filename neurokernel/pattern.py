#!/usr/bin/env python

"""
Represent connectivity pattern using pandas DataFrame.
"""

from collections import OrderedDict
import itertools
import re

from future.utils import iteritems
from past.builtins import basestring
import networkx as nx
import numpy as np
import pandas as pd

from .plsel import Selector, SelectorMethods
from .pm import BasePortMapper

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
    plsel.SelectorMethods
    """

    def __init__(self, selector='', columns=['interface', 'io', 'type']):

        # All ports in an interface must contain at least the following
        # attributes:
        assert set(columns).issuperset(['interface', 'io', 'type'])
        self.sel = SelectorMethods()
        assert not(self.sel.is_ambiguous(selector))
        self.num_levels = self.sel.max_levels(selector)
        names = [i for i in range(self.num_levels)]
        idx = self.sel.make_index(selector, names)
        self.__validate_index__(idx)
        self.data = pd.DataFrame(index=idx, columns=columns, dtype=object)

        # Dictionary containing mappers for different port types:
        self.pm = {}

    def __validate_index__(self, idx):
        """
        Raise an exception if the specified index will result in an invalid interface.
        """

        if idx.duplicated().any():
            raise ValueError('Duplicate interface index entries detected.')

    def __getitem__(self, key):
        if type(key) == tuple and len(key) > 1:
            return self.sel.select(self.data[list(key[1:])], key[0])
        else:
            return self.sel.select(self.data, key)

    def __setitem__ambiguous__(self, key, value):
        if type(key) == tuple:
            selector = key[0]
        else:
            selector = key

        # Ensure that the specified selector can actually be used against the
        # Interface's internal DataFrame:
        try:
            idx = self.sel.get_index(self.data, selector,
                                     names=self.data.index.names)
        except ValueError:
            raise ValueError('cannot create index with '
                             'selector %s and column names %s' \
                             % (selector, str(self.data.index.names)))

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

        for k, v in iteritems(data):
            self.data[k].loc[idx] = v

    def __setitem__(self, key, value):
        if type(key) == tuple:
            selector = key[0]
        else:
            selector = key

        # Fall back to slower method if the selector is ambiguous:
        if self.sel.is_ambiguous(selector):
            self.__setitem__ambiguous__(key, value)
            return
        else:
            selector = Selector(selector)

        # Don't waste time trying to do anything if the selector is empty:
        if not selector.nonempty:
            return

        # If the number of specified identifiers doesn't exceed the size of the
        # data array, enlargement by specifying identifiers that are not in
        # the index will not occur:
        assert len(selector) <= len(self.data)

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

        if selector.max_levels == 1:
            s = [i for i in itertools.chain(*selector.expanded)]
        else:
            s = self.sel.pad_selector(selector.expanded,
                                      len(self.index.levshape))
        for k, v in iteritems(data):
            self.data[k].loc[s] = v

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

    @property
    def idx_levels(self):
        """
        Number of levels in Interface index.
        """

        if isinstance(self.data.index, pd.MultiIndex):
            return len(self.index.levels)
        else:
            return 1

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
        >>> import plsel, pattern
        >>> import pandas
        >>> idx = plsel.SelectorMethods.make_index('/foo[0:2]')
        >>> data = [[0, 'in', 'spike'], [1, 'out', 'gpot']]
        >>> columns = ['interface', 'io', 'type']
        >>> df = pandas.DataFrame(data, index=idx, columns=columns)
        >>> i = pattern.Interface.from_df(df)

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
            if len(df.index):
                i = cls(df.index.tolist(), df.columns)
            else:
                i = cls([()], df.columns)
        elif isinstance(df.index, pd.Index):
            if len(df.index):
                i = cls([(s,) for s in df.index.tolist()], df.columns)
            else:
                i = cls([()], df.columns)
        else:
            raise ValueError('invalid index type')
        i.data = df.copy()
        i.__validate_index__(i.index)
        return i

    @classmethod
    def from_csv(cls, file_name, **kwargs):
        """
        Create an Interface from a properly formatted CSV file.

        Parameters
        ----------
        file_name : str
            File name of CSV file containing interface data.
        kwargs : dict
            Options to pass to `DataFrame.from_csv()`

        Returns
        -------
        i : Interface
            Generated Interface instance.
        """

        df = pd.DataFrame.from_csv(file_name, **kwargs)
        return cls.from_df(df)

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
        for k, v in iteritems(d):
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
        return cls.from_dict(g.nodes)

    @classmethod
    def from_selectors(cls, sel, sel_in='', sel_out='',
                       sel_spike='', sel_gpot='', *sel_int_list):
        """
        Create an Interface instance from selectors.

        Parameters
        ----------
        sel : str, unicode, or sequence
            Selector describing all ports comprised by interface.
        sel_in : str, unicode, or sequence
            Selector describing the interface's input ports.
        sel_out : str, unicode, or sequence
            Selector describing the interface's output ports.
        sel_spike : str, unicode, or sequence
            Selector describing the interface's spiking ports.
        sel_gpot : str, unicode, or sequence
            Selector describing the interface's graded potential ports.
        sel_int_list : list of str, unicode, or sequence
            Selectors consecutively describing the ports associated with interface 0,
            interface 1, etc.

        Returns
        -------
        i : Interface
            Generated interface instance.
        """

        i = cls(sel)
        i[sel_in, 'io'] = 'in'
        i[sel_out, 'io'] = 'out'
        i[sel_spike, 'type'] = 'spike'
        i[sel_gpot, 'type'] = 'gpot'
        for n, sel_int in enumerate(sel_int_list):
            i[sel_int, 'interface'] = n
        return i

    def gpot_ports(self, i=None, tuples=False):
        """
        Restrict Interface ports to graded potential ports.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return all graded potential ports.
        tuples : bool
            If True, return a list of tuples; if False, return an
            Interface instance.

        Returns
        -------
        interface : Interface or list of tuples
            Either an Interface instance containing all graded potential ports and
            their attributes in the specified interface, or a list of tuples
            corresponding to the expanded ports.
        """

        if i is None:
            try:
                df = self.data[self.data['type'] == 'gpot']
            except:
                df = None
        else:
            try:
                df = self.data[(self.data['type'] == 'gpot') & \
                               (self.data['interface'] == i)]
            except:
                df = None
        if tuples:
            if df is None:
                return []
            else:
                return df.index.tolist()
        else:
            if df is None:
                return Interface()
            else:
                return self.from_df(df)

    def in_ports(self, i=None, tuples=False):
        """
        Restrict Interface ports to input ports.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return all input ports.
        tuples : bool
            If True, return a list of tuples; if False, return an
            Interface instance.

        Returns
        -------
        interface : Interface or list of tuples
            Either an Interface instance containing all input ports and
            their attributes in the specified interface, or a list of tuples
            corresponding to the expanded ports.
        """

        if i is None:
            try:
                df = self.data[self.data['io'] == 'in']
            except:
                df = None
        else:
            try:
                df = self.data[(self.data['io'] == 'in') & \
                               (self.data['interface'] == i)]
            except:
                df = None
        if tuples:
            if df is None:
                return []
            else:
                return df.index.tolist()
        else:
            if df is None:
                return Interface()
            else:
                return self.from_df(df)

    def interface_ports(self, i=None, tuples=False):
        """
        Restrict Interface ports to specific interface.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return all ports.
        tuples : bool
            If True, return a list of tuples; if False, return an
            Interface instance.

        Returns
        -------
        interface : Interface
            Either an Interface instance containing all ports and
            their attributes in the specified interface, or a list of tuples
            corresponding to the expanded ports.
        """

        if i is None:
            if tuples:
                return self.index.tolist()
            else:
                return self.copy()
        else:
            try:
                df = self.data[self.data['interface'] == i]
            except:
                df = None
            if tuples:
                if df is None:
                    return []
                else:
                    return df.index.tolist()
            else:
                if df is None:
                    return Interface()
                else:
                    return self.from_df(df)

    def _merge_on_interfaces(self, a, i, b):
        """
        Merge contents of this and another Interface instance.

        Notes
        -----
        If the number of levels in one Interface instance's DataFrame index is
        greater than that of the other, the number of levels in the index of the
        merged DataFrames instances is set to the former and the index with the
        smaller number is padded with blank entries to enable Panda's merge
        mechanism to function properly.
        """

        assert isinstance(i, Interface)
        df_left = self.data[self.data['interface'] == a]
        df_right = i.data[i.data['interface'] == b]
        n_left_names = len(self.data.index.names)
        n_right_names = len(i.data.index.names)

        # Pandas' merge mechanism fails if the number of levels in each of the
        # merged MultiIndex indices differs and there is overlap of more than
        # one level; we therefore pad the index with the smaller number of
        # levels before attempting the merge:
        if n_left_names > n_right_names:
            for n in range(i.num_levels, i.num_levels+(n_left_names-n_right_names)):
                new_col = str(n)
                df_right[new_col] = ''
                df_right.set_index(new_col, append=True, inplace=True)
        elif n_left_names < n_right_names:
            for n in range(self.num_levels, self.num_levels+(n_right_names-n_left_names)):
                new_col = str(n)
                df_left[new_col] = ''
                df_left.set_index(new_col, append=True, inplace=True)
        return pd.merge(df_left, df_right,
                        left_index=True,
                        right_index=True)

    def get_common_ports(self, a, i, b, t=None):
        """
        Get port identifiers common to this and another Interface instance.

        Parameters
        ----------
        a : int
            Identifier of interface in the current instance.
        i : Interface
            Interface instance containing the other interface.
        b : int
            Identifier of interface in instance `i`.
        t : str or unicode
            If not None, restrict output to those identifiers with the specified
            port type.

        Returns
        -------
        result : list of tuple
            Expanded port identifiers shared by the two specified Interface
            instances.

        Notes
        -----
        The number of levels of the returned port identifiers is equal to the
        maximum number of levels of this Interface instance.

        The order of the returned port identifiers is not guaranteed.
        """

        if t is None:
            x = self.data[self.data['interface'] == a]
            y = i.data[i.data['interface'] == b]
        else:
            x = self.data[(self.data['interface'] == a) & (self.data['type'] == t)]
            y = i.data[(i.data['interface'] == b) & (i.data['type'] == t)]
        if isinstance(x.index, pd.MultiIndex):
            x_list = [tuple(a for a in b if a != '') \
                      for b in x.index]
        else:
            x_list = [(a,) for a in x.index]
        if isinstance(y.index, pd.MultiIndex):
            y_list = [tuple(a for a in b if a != '') \
                      for b in y.index]
        else:
            y_list = [(a,) for a in y.index]

        return list(set(x_list).intersection(y_list))

    def is_compatible(self, a, i, b, allow_subsets=False):
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
        allow_subsets : bool
            If True, interfaces that contain a compatible subset of ports are
            deemed to be compatible; otherwise, all ports in the two interfaces
            must be compatible.

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

        # Merge the interface data on their indices (i.e., their port identifiers):
        data_merged = self._merge_on_interfaces(a, i, b)

        # Check whether there are compatible subsets, i.e., at least one pair of
        # ports from the two interfaces that are compatible with each other:
        if allow_subsets:

            # If the interfaces share no identical port identifiers, they are
            # incompatible:
            if not len(data_merged):
                return False

            # Compatible identifiers must have the same non-null 'type'
            # attribute and their non-null 'io' attributes must be the inverse
            # of each other:
            if not data_merged.apply(lambda row: \
                    ((row['type_x'] == row['type_y']) or \
                     (pd.isnull(row['type_x']) and pd.isnull(row['type_y']))) and \
                    ((row['io_x'] == 'out' and row['io_y'] == 'in') or \
                     (row['io_x'] == 'in' and row['io_y'] == 'out') or \
                     (pd.isnull(row['io_x']) and pd.isnull(row['io_y']))),
                                     axis=1).any():
                return False

        # Require that all ports in the two interfaces be compatible:
        else:

            # If one interface contains identifiers not in the other, they are
            # incompatible:
            if len(data_merged) < max(len(self.data[self.data['interface'] == a]),
                                      len(i.data[i.data['interface'] == b])):
                return False

            # Compatible identifiers must have the same non-null 'type'
            # attribute and their non-null 'io' attributes must be the inverse
            # of each other:
            if not data_merged.apply(lambda row: \
                    ((row['type_x'] == row['type_y']) or \
                     (pd.isnull(row['type_x']) and pd.isnull(row['type_y']))) and \
                    ((row['io_x'] == 'out' and row['io_y'] == 'in') or \
                     (row['io_x'] == 'in' and row['io_y'] == 'out') or \
                     (pd.isnull(row['io_x']) and pd.isnull(row['io_y']))),
                                     axis=1).all():
                return False

        # All tests passed:
        return True

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

        try:
            # Pad the expanded selector with blanks to prevent pandas from
            # spurious matches such as mistakenly validating '/foo' as being in
            # an Interface that only contains the ports '/foo[0:2]':
            idx = self.sel.expand(s, self.idx_levels)
            if not isinstance(self.data.index, pd.MultiIndex):
                idx = [x[0] for x in idx]
            d = self.data['interface'].loc[idx]

            if isinstance(d, int):
                return True
            if np.any(d.isnull().tolist()):
                return False
            else:
                return True
        except:
            return self.sel.is_in(s, self.index.tolist())

    def out_ports(self, i=None, tuples=False):
        """
        Restrict Interface ports to output ports.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return all output ports.
        tuples : bool
            If True, return a list of tuples; if False, return an
            Interface instance.

        Returns
        -------
        interface : Interface or list of tuples
            Either an Interface instance containing all output ports and
            their attributes in the specified interface, or a list of tuples
            corresponding to the expanded ports.
        """

        if i is None:
            try:
                df = self.data[self.data['io'] == 'out']
            except:
                df = None
        else:
            try:
                df = self.data[(self.data['io'] == 'out') & \
                               (self.data['interface'] == i)]
            except:
                df = None
        if tuples:
            if df is None:
                return []
            else:
                return df.index.tolist()
        else:
            if df is None:
                return Interface()
            else:
                return self.from_df(df)

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

    def spike_ports(self, i=None, tuples=False):
        """
        Restrict Interface ports to spiking ports.

        Parameters
        ----------
        i : int
            Interface identifier. If None, return all spiking ports.
        tuples : bool
            If True, return a list of tuples; if False, return an
            Interface instance.

        Returns
        -------
        interface : Interface or list of tuples
            Either an Interface instance containing all spiking ports and
            their attributes in the specified interface, or a list of tuples
            corresponding to the expanded ports.
        """

        if i is None:
            try:
                df = self.data[self.data['type'] == 'spike']
            except:
                df = None
        else:
            try:
                df = self.data[(self.data['type'] == 'spike') & \
                               (self.data['interface'] == i)]
            except:
                df = None
        if tuples:
            if df is None:
                return []
            else:
                return df.index.tolist()
        else:
            if df is None:
                return Interface()
            else:
                return self.from_df(df)

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
                if isinstance(s, basestring):
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
            idx = self.sel.expand(s, self.idx_levels)
            if not isinstance(self.data.index, pd.MultiIndex):
                idx = [x[0] for x in idx]
            d = self.data['interface'].loc[idx]
            s = set(d)
            s.discard(np.nan)
            return s
        except:
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

    def set_pm(self, t, pm):
        """
        Set port mapper associated with a specific port type.

        Parameters
        ----------
        t : str or unicode
            Port type.
        pm : neurokernel.plsel.BasePortMapper
            Port mapper to save.
        """

        # Ensure that the ports in the specified port mapper are a subset of
        # those in the interface associated with the specified type:
        assert isinstance(pm, BasePortMapper)
        if not self.sel.is_in(pm.index.tolist(),
                              self.pm[t].index.tolist()):
            raise ValueError('cannot set mapper using undefined selectors')
        self.pm[t] = pm.copy()

    def equals(self, other):
        """
        Check whether this interface is equivalent to another interface.

        Parameters
        ----------
        other : neurokernel.pattern.Interface
            Interface instance to compare to this Interface.

        Returns
        -------
        result : bool
            True if the interfaces are identical.

        Notes
        -----
        Interfaces containing the same rows in different orders are not
        regarded as equivalent.
        """

        assert isinstance(other, Interface)
        return self.data.equals(other.data)

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
    plsel.SelectorMethods
    """

    def __init__(self, *selectors, **kwargs):
        columns = kwargs.get('columns', ['conn'])
        self.sel = SelectorMethods()

        # Force sets of identifiers to be disjoint so that no identifier can
        # denote a port in more than one set:
        assert self.sel.are_disjoint(*selectors)

        # Collect all of the selectors:
        selector = []
        for s in selectors:
            if isinstance(s, Selector) and len(s) != 0:
                selector.extend(s.expanded)
            elif isinstance(s, basestring):
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
        names = ['from_%s' % i for i in range(self.num_levels['from'])]+ \
                ['to_%s' %i for i in range(self.num_levels['to'])]
        levels = [[] for i in range(len(names))]
        labels = [[] for i in range(len(names))]
        idx = pd.MultiIndex(levels=levels, codes=labels, names=names)

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
            by one selector may be in any other selector, and non-empty.
        from_sel, to_sel : str
            Selectors that describe the pattern's initial index. If specified,
            both selectors must be set. If no selectors are set, the index is
            initially empty.
        gpot_sel, spike_sel : str
            Selectors that describe the graded potential and spiking ports in a
            pattern's initial index.
        data : numpy.ndarray, dict, or pandas.DataFrame
            Data to load store in class instance.
        columns : sequence of str
            Data column names.
        comp_op : str
            Operator to use to combine selectors into single selector that
            comprises both the source and destination ports in a pattern.
        validate : bool
            If True, validate the index of the Pattern's DataFrame.

        Returns
        -------
        result : Pattern
            Pattern instance.
        """

        from_sel = kwargs.get('from_sel', None)
        to_sel = kwargs.get('to_sel', None)
        gpot_sel = kwargs.get('gpot_sel', None)
        spike_sel = kwargs.get('spike_sel', None)
        data = kwargs.get('data', None)
        columns = kwargs.get('columns', ['conn'])
        comb_op = kwargs.get('comb_op', '+')
        validate = kwargs.get('validate', True)

        # Create empty pattern:
        for s in selectors:
            if not len(s):
                raise ValueError('cannot create pattern with empty selector %s' % s)
        p = cls(*selectors, columns=columns)

        # Construct index from concatenated selectors if specified:
        names = p.data.index.names
        if (from_sel is None and to_sel is None):
            levels = [[] for i in range(len(names))]
            labels = [[] for i in range(len(names))]
            idx = pd.MultiIndex(levels=levels, codes=labels, names=names)
        elif isinstance(from_sel, Selector) and isinstance(to_sel, Selector):
            if comb_op == '.+':
                idx = p.sel.make_index(Selector.concat(from_sel, to_sel), names)
            elif comb_op == '+':
                idx = p.sel.make_index(Selector.prod(from_sel, to_sel), names)
            else:
                raise ValueError('incompatible selectors specified')
        else:
            idx = p.sel.make_index('(%s)%s(%s)' % (from_sel, comb_op, to_sel), names)
        if validate:
            p.__validate_index__(idx)

        # Replace the pattern's DataFrame:
        p.data = pd.DataFrame(data=data, index=idx, columns=columns, dtype=object)

        # Update the `io` attributes of the pattern's interfaces:
        p.interface[from_sel, 'io'] = 'in'
        p.interface[to_sel, 'io'] = 'out'

        # Update the `type` attributes of the pattern's interface:
        if gpot_sel is not None:
            p.interface[gpot_sel, 'type'] = 'gpot'
        if spike_sel is not None:
            p.interface[spike_sel, 'type'] = 'spike'

        return p

    def clear(self):
        """
        Clear all connections in class instance.
        """

        self.interface.clear()
        self.data.drop(self.data.index, inplace=True)

    @classmethod
    def from_df(cls, df_int, df_pat):
        """
        Create a Pattern from properly formatted DataFrames.

        Parameters
        ----------
        df_int : pandas.DataFrame
            DataFrame with a MultiIndex and data columns 'interface',
            'io', and 'type' (additional columns may also be present) that
            describes the pattern's interfaces. The index's rows must correspond
            to individual port identifiers.
        df_pat : pandas.DataFrame
            DataFrame with a MultiIndex and a data column 'conn' (additional
            columns may also be present) that describes the connections between
            ports in the pattern's interfaces. The index's level names must be
            'from_0'..'from_N', 'to_0'..'to_M', where N and M are the maximum
            number of levels in the pattern's two interfaces.
        """

        # Create pattern with phony selectors:
        pat = cls('/foo[0]', '/bar[0]')

        # Check that the 'interface' column of the interface DataFrame is set:
        if any(df_int['interface'].isnull()):
            raise ValueError('interface column must be set')

        # Create interface:
        pat.interface = Interface.from_df(df_int)

        # The pattern DataFrame's index must contain at least two levels:
        assert isinstance(df_pat.index, pd.MultiIndex)

        # Check that pattern DataFrame index levels are named correctly,
        # i.e., from_0..from_N and to_0..to_N, where N is equal to
        # pat.interface.num_levels:
        num_levels = pat.interface.num_levels
        if df_pat.index.names != ['from_%i' % i for i in range(num_levels)]+\
           ['to_%i' % i for i in range(num_levels)]:
            raise ValueError('incorrectly named pattern index levels')

        for t in df_pat.index.tolist():
            from_t = t[0:num_levels]
            to_t = t[num_levels:2*num_levels]
            if from_t not in df_int.index or to_t not in df_int.index:
                raise ValueError('pattern DataFrame contains identifiers '
                                 'not in interface DataFrame')

        pat.data = df_pat.copy()
        return pat

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
            Selectors that describe the pattern's initial index. If specified,
            both selectors must be set; the 'io' attribute of the ports
            comprised by these selectors is respectively set to 'out' and
            'in'. If no selectors are set, the index is initially empty.
        gpot_sel, spike_sel : str
            Selectors that describe the graded potential and spiking ports in a
            pattern's initial index. If specified, the 'type' attribute of the
            ports comprised by these selectors is respectively set to 'gpot'
            and 'spike'.
        data : numpy.ndarray, dict, or pandas.DataFrame
            Data to load store in class instance.
        columns : sequence of str
            Data column names.
        validate : bool
            If True, validate the index of the Pattern's DataFrame.

        Returns
        -------
        result : Pattern
            Pattern instance.
        """

        from_sel = kwargs.get('from_sel', None)
        to_sel = kwargs.get('to_sel', None)
        gpot_sel = kwargs.get('gpot_sel', None)
        spike_sel = kwargs.get('spike_sel', None)
        data = kwargs.get('data', None)
        columns = kwargs.get('columns', ['conn'])
        validate = kwargs.get('validate', True)
        return cls._create_from(*selectors, from_sel=from_sel, to_sel=to_sel,
                                gpot_sel=gpot_sel, spike_sel=spike_sel,
                                data=data, columns=columns, comb_op='+', validate=validate)

    def gpot_ports(self, i=None, tuples=False):
        return self.interface.gpot_ports(i, tuples)
    gpot_ports.__doc__ = Interface.gpot_ports.__doc__

    def in_ports(self, i=None, tuples=False):
        return self.interface.in_ports(i, tuples)
    in_ports.__doc__ = Interface.in_ports.__doc__

    def interface_ports(self, i=None, tuples=False):
        return self.interface.interface_ports(i, tuples)
    interface_ports.__doc__ = Interface.interface_ports.__doc__

    def out_ports(self, i=None, tuples=False):
        return self.interface.out_ports(i, tuples)
    out_ports.__doc__ = Interface.out_ports.__doc__

    def spike_ports(self, i=None, tuples=False):
        return self.interface.spike_ports(i, tuples)
    spike_ports.__doc__ = Interface.spike_ports.__doc__

    def connected_ports(self, i=None, tuples=False):
        """
        Return ports that are connected by the pattern.

        Parameters
        ----------
        i : int
            Interface identifier.
        tuples : bool
            If True, return a list of tuples; if False, return an
            Interface instance.

        Returns
        -------
        interface : Interface
            Either an Interface instance containing all connected ports and
            their attributes in the specified interface, or a list of tuples
            corresponding to the expanded ports.

        Notes
        -----
        Returned ports are listed in lexicographic order.
        """

        # Use sets to accumulate the expanded ports to avoid passing duplicates
        # to DataFrame.ix.__getitem__():
        ports = set()
        for t in self.data.index:
            ports.add(t[0:self.num_levels['from']])
            ports.add(t[self.num_levels['from']:self.num_levels['from']+self.num_levels['to']])

        # Sort the expanded ports so that the results are returned in
        # lexicographic order:
        df = self.interface.data.loc[sorted(ports)]
        if i is None:
            if tuples:
                return df.index.tolist()
            else:
                return Interface.from_df(df)
        else:
            if tuples:
                return df[df['interface'] == i].index.tolist()
            else:
                return Interface.from_df(df[df['interface'] == i])

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
            both selectors must be set; the 'io' attribute of the ports
            comprised by these selectors is respectively set to 'out' and
            'in'. If no selectors are set, the index is initially empty.
        gpot_sel, spike_sel : str
            Selectors that describe the graded potential and spiking ports in a
            pattern's initial index. If specified, the 'type' attribute of the
            ports comprised by these selectors is respectively set to 'gpot'
            and 'spike'.
        columns : sequence of str
            Data column names.
        validate : bool
            If True, validate the index of the Pattern's DataFrame.

        Returns
        -------
        result : Pattern
            Pattern instance.
        """

        from_sel = kwargs.get('from_sel', None)
        to_sel = kwargs.get('to_sel', None)
        gpot_sel = kwargs.get('gpot_sel', None)
        spike_sel = kwargs.get('spike_sel', None)
        data = kwargs.get('data', None)
        columns = kwargs.get('columns', ['conn'])
        validate = kwargs.get('validate', True)
        return cls._create_from(*selectors, from_sel=from_sel, to_sel=to_sel,
                                gpot_sel=gpot_sel, spike_sel=spike_sel,
                                data=data, columns=columns, comb_op='.+', validate=validate)

    def __validate_index__(self, idx):
        """
        Raise an exception if the specified index will result in an invalid pattern.
        """

        # Prohibit duplicate connections:
        if idx.duplicated().any():
            raise ValueError('Duplicate pattern entries detected.')

        # Prohibit fan-in connections (i.e., patterns whose index has duplicate
        # 'from' port identifiers):
        from_idx, to_idx = self.split_multiindex(idx,
                                                 self.from_slice, self.to_slice)
        if to_idx.duplicated().any():
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
        return self.interface.is_in_interfaces(selector)
        if len(self.interface[selector]) > 0:
            return True
        else:
            return False

    def connected_port_pairs(self, as_str=False):
        """
        Return connections as pairs of port identifiers.

        Parameters
        ----------
        as_str : bool
            If True, return connections as a list of identifier
            string pairs. Otherwise, return them as pairs of token tuples.
        """

        if as_str:
            return [(self.sel.tokens_to_str(row[self.from_slice]),
                     self.sel.tokens_to_str(row[self.to_slice])) \
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

        # Expand and pad the specified 'from' and 'to' selectors:
        key_0_exp = self.sel.expand(key[0], self.num_levels['from'])
        key_1_exp = self.sel.expand(key[1], self.num_levels['to'])

        # Concatenate the selectors:
        selector = tuple(tuple(j for j in itertools.chain(*i)) \
                for i in itertools.product(key_0_exp, key_1_exp))

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
            for k, v in iteritems(data):
                self.data[k].loc[idx] = v

        # Otherwise, populate a new DataFrame with the specified attributes:
        else:
            new_data = self.data.append(pd.DataFrame(data=data, index=idx,
                                                     dtype=object))

            # Validate updated DataFrame's index before updating the instance's
            # data attribute:
            self.__validate_index__(new_data.index)
            self.data = new_data
            self.data.sort_index(inplace=True)

        # Update the `io` attributes of the pattern's interfaces:
        self.interface[key[0], 'io'] = 'in'
        self.interface[key[1], 'io'] = 'out'

    def __getitem__(self, key):
        assert len(key) >= 2
        sel_0 = self.sel.expand(key[0])
        sel_1 = self.sel.expand(key[1])
        selector = [f+t for f, t in itertools.product(sel_0, sel_1)]
        if len(key) > 2:
            return self.sel.select(self.data[list(key[2:])], selector=selector)
        else:
            return self.sel.select(self.data, selector=selector)

    def src_idx(self, src_int, dest_int,
                src_type=None, dest_type=None, dest_ports=None, duplicates=False):
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
        duplicates : bool
            If True, include duplicate ports in output.

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
            if dest_type == 'gpot':
                to_int = self.interface.gpot_ports(dest_int)
            elif dest_type == 'spike':
                to_int = self.interface.spike_ports(dest_int)
            else:
                to_f = lambda x: x['type'] == dest_type
                to_int = self.interface.interface_ports(dest_int).data_select(to_f)

        # Filter destination ports by specified ports:
        if dest_ports is None:
            to_idx = set(to_int.index)
        else:
            to_idx = set(to_int[dest_ports].index)

        # Filter source ports by specified type:
        if src_type is None:
            from_int = self.interface.interface_ports(src_int)
        else:
            if src_type == 'gpot':
                from_int = self.interface.gpot_ports(src_int)
            elif src_type == 'spike':
                from_int = self.interface.spike_ports(src_int)
            else:
                from_f = lambda x: x['type'] == src_type
                from_int = self.interface.interface_ports(src_int).data_select(from_f)


        from_idx = set(from_int.index)

        idx = []
        for x in self.data.index:
            tmp1 = x[self.from_slice]
            if tmp1 in from_idx:
                if x[self.to_slice] in to_idx:
                    idx.append(tmp1)

        if not duplicates:
            # Remove duplicate tuples from output without perturbing the order
            # of the remaining tuples:
            return list(OrderedDict.fromkeys(idx).keys())
        else:
            return idx

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

        Notes
        -----
        No `duplicates` parameter is provided because fan-in from multiple
        source ports to a single destination port is not permitted.
        """

        assert src_int != dest_int
        assert src_int in self.interface.interface_ids and \
            dest_int in self.interface.interface_ids

        # Filter source ports by specified type:
        if src_type is None:
            from_int = self.interface.interface_ports(src_int)
        else:
            if src_type == 'gpot':
                from_int = self.interface.gpot_ports(src_int)
            elif dest_type == 'spike':
                from_int = self.interface.spike_ports(src_int)
            else:
                from_f = lambda x: x['type'] == src_type
                from_int = self.interface.interface_ports(src_int).data_select(from_f)

        # Filter source ports by specified ports:
        if src_ports is None:
            from_idx = set(from_int.index)
        else:
            from_idx = set(from_int[src_ports].index)

        # Filter destination ports by specified type:
        if dest_type is None:
            to_int = self.interface.interface_ports(dest_int)
        else:
            if dest_type == 'gpot':
                to_int = self.interface.gpot_ports(dest_int)
            elif dest_type == 'spike':
                to_int = self.interface.spike_ports(dest_int)
            else:
                to_f = lambda x: x['type'] == dest_type
                to_int = self.interface.interface_ports(dest_int).data_select(to_f)

        to_idx = set(to_int.index)

        idx = []
        for x in self.data.index:
            tmp1 = x[self.to_slice]
            if tmp1 in to_idx:
                if x[self.from_slice] in from_idx:
                    idx.append(tmp1)
        # Remove duplicate tuples from output without perturbing the order
        # of the remaining tuples:
        return list(OrderedDict.fromkeys(idx).keys())

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
            True if at least one connection from a port identifier in interface
            `from_int` to a port identifier in interface `to_int` exists.
        """

        assert from_int != to_int
        assert from_int in self.interface.interface_ids
        assert to_int in self.interface.interface_ids

        # Get indices of the 'from' and 'to' interfaces as lists to speed up the
        # check below [*]:
        from_idx = set(self.interface.data[self.interface.data['interface'] == from_int].index.tolist())
        to_idx = set(self.interface.data[self.interface.data['interface'] == to_int].index.tolist())

        # Get index of all defined connections:
        idx = self.data[self.data['conn'] != 0].index
        for t in idx:

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

            # Check whether port identifiers are in the interface indices [*]:
            if from_id in from_idx and to_id in to_idx:
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
    def from_graph(cls, g, return_key_order = False):
        """Convert a NetworkX directed graph into a Pattern instance.

        Parameters
        ----------
        g : networkx.DiGraph
            Graph to convert. The node identifiers must be port identifiers.

        return_key_order : bool
            Whether to return the keys of all identifiers

        Returns
        -------
        p : Pattern
            Pattern instance.
        key : List
              A list of keys of identifier, of which the order determines
              the numbering of interfaces

        Notes
        -----
        The nodes in the specified graph must contain an 'interface' attribute.

        Port attributes other than 'interface', 'io', and 'type' are not stored
        in the created Pattern instance's interface.
        """

        assert type(g) == nx.DiGraph

        # Group port identifiers by interface number and whether the ports are
        # graded potential, or spiking:
        ports_by_int = {}
        ports_gpot = []
        ports_spike = []
        ports_from = []
        ports_to = []
        for n, data in g.nodes(data=True):
            assert SelectorMethods.is_identifier(n)
            assert 'interface' in data
            if not data['interface'] in ports_by_int:
                ports_by_int[data['interface']] = []
            ports_by_int[data['interface']].append(n)

            if 'type' in data:
                if data['type'] == 'gpot':
                    ports_gpot.append(n)
                elif data['type'] == 'spike':
                    ports_spike.append(n)

        # Use connection direction to determine whether ports are source or
        # destination (XXX should this check whether the io attributes are
        # consistent with the connection directions?):
        for f, t in g.edges():
            ports_from.append(f)
            ports_to.append(t)

        # Create selectors for each interface number:
        selector_list = []
        key_order = sorted(ports_by_int.keys())
        for interface in key_order:
            selector_list.append(','.join(ports_by_int[interface]))

        p = cls.from_concat(*selector_list,
                            from_sel=','.join(ports_from),
                            to_sel=','.join(ports_to),
                            gpot_sel=','.join(ports_gpot),
                            spike_sel=','.join(ports_spike),
                            data=1)

        p.data.sort_index(inplace=True)
        p.interface.data.sort_index(inplace=True)

        if return_key_order:
            return p, key_order
        else:
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
            if not isinstance(self.interface.data.index,
                              pd.MultiIndex):
                t = (t,)
            id = self.sel.tokens_to_str(t)

            # Replace NaNs with empty strings:
            d = {k: (v if str(v) != 'nan' else '') \
                 for k, v in iteritems(self.interface.data.loc[t].to_dict())}

            # Each node's name corresponds to the port identifier string:
            g.add_node(id, **d)

        # Add all of the connections as edges:
        for t in self.data.index:
            t_from = t[self.from_slice]
            t_to = t[self.to_slice]
            id_from = self.sel.tokens_to_str(t_from)
            id_to = self.sel.tokens_to_str(t_to)
            d = self.data.loc[t].to_dict()

            # Discard the 'conn' attribute because the existence of the edge
            # indicates that the connection exists:
            if 'conn' in d:
                d.pop('conn')

            g.add_edge(id_from, id_to, **d)

        return g

def are_compatible(sel_in_0, sel_out_0, sel_spike_0, sel_gpot_0,
                   sel_in_1, sel_out_1, sel_spike_1, sel_gpot_1,
                   allow_subsets=False):
    """
    Check whether two interfaces specified as selectors can be connected.

    Parameters
    ----------
    sel_in_0, sel_out_0, sel_spike_0, sel_gpot_0 : Selector, str, unicode
        Input, output, spiking, and graded potential ports in first interface.
    sel_in_0, sel_out_0, sel_spike_0, sel_gpot_0 : Selector, str, unicode
        Input, output, spiking, and graded potential ports in second interface.
    allow_subsets : bool
        If True, interfaces that contain a compatible subset of ports are
        deemed to be compatible; otherwise, all ports in the two interfaces
        must be compatible.

    Results
    -------
    result : bool
        True if interfaces are compatible, False otherwise.
    """

    sel_in_0 = Selector(sel_in_0)
    sel_out_0 = Selector(sel_out_0)
    sel_spike_0 = Selector(sel_spike_0)
    sel_gpot_0 = Selector(sel_gpot_0)
    sel_0 = Selector.union(sel_in_0, sel_out_0, sel_spike_0, sel_gpot_0)

    sel_in_1 = Selector(sel_in_1)
    sel_out_1 = Selector(sel_out_1)
    sel_spike_1 = Selector(sel_spike_1)
    sel_gpot_1 = Selector(sel_gpot_1)
    sel_1 = Selector.union(sel_in_1, sel_out_1, sel_spike_1, sel_gpot_1)

    int_0 = Interface.from_selectors(sel_0, sel_in_0, sel_out_0,
                                     sel_spike_0, sel_gpot_0, sel_0)
    int_1 = Interface.from_selectors(sel_1, sel_in_1, sel_out_1,
                                     sel_spike_1, sel_gpot_1, sel_1)

    return int_0.is_compatible(0, int_1, 0)
