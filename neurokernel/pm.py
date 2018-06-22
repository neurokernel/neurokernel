#!/usr/bin/env python

"""
Port mapper classes.
"""

import numpy as np
import pandas as pd

from .plsel import SelectorMethods

class BasePortMapper(object):
    """
    Maps integer sequence to/from path-like port identifiers.

    Examples
    --------
    >>> pm = BasePortMapper('/[a,b][0:2]')
    >>> print pm.ports_to_inds('/b[0:2]')
    array([2, 3])
    >>> print pm.inds_to_ports([0, 1])
    [('a', 0), ('a', 1)]

    Parameters
    ----------
    selector : str, unicode, or sequence
        Selector string (e.g., '/foo[0:2]') or sequence of token sequences
        (e.g., [['foo', (0, 2)]]) to map to `data`.
    portmap : sequence of int
        Integer indices to map to port identifiers. If no map is specified,
        it is assumed to be an array of consecutive integers from 0
        through one less than the number of ports.

    Attributes
    ----------
    index : pandas.MultiIndex
        Index of port identifiers.
    portmap : pandas.Series
        Map of port identifiers to integer indices.

    Notes
    -----
    The selectors may not contain any '*' or '[:]' characters.
    A single port identifier may be mapped to multiple integer indices,
    but not vice-versa.
    """

    def __init__(self, selector, portmap=None):
        self.sel = SelectorMethods()
        N = self.sel.count_ports(selector)
        if portmap is None:
            self.portmap = pd.Series(data=np.arange(N))
        else:
            assert len(portmap) == N
            self.portmap = pd.Series(data=np.array(portmap))
        self.portmap.index = self.sel.make_index(selector)

    def copy(self):
        """
        Return copy of this port mapper.

        Returns
        -------
        result : neurokernel.plsel.BasePortMapper
            Copy of port mapper instance.
        """

        c = BasePortMapper('')
        c.portmap = self.portmap.copy()
        return c

    @classmethod
    def from_index(cls, idx, portmap=None):
        """
        Create port mapper from a Pandas index and a sequence of integer indices.

        Parameters
        ----------
        index : pandas.MultiIndex
            Index containing selector data.
        portmap : sequence of int
            Integer indices to map to port identifiers. If no map is specified,
            it is assumed to be an array of consecutive integers from 0
            through one less than the number of ports.

        Returns
        -------
        result : neurokernel.plsel.BasePortMapper
            New port mapper instance.

        Notes
        -----
        If specified, the portmap sequence is copied into the new mapper to avoid
        side effects associated with modifying the specified sequence after
        mapper instantiation.
        """

        pm = cls('')
        N = len(idx)
        if portmap is None:
            pm.portmap = pd.Series(np.arange(N), idx)
        else:
            assert len(portmap) == N
            pm.portmap = pd.Series(np.array(portmap), idx)
        return pm

    @classmethod
    def from_pm(cls, pm):
        """
        Create a new port mapper instance given an existing instance.

        Parameters
        ----------
        result : neurokernel.plsel.BasePortMapper
            Existing port mapper instance.

        Returns
        -------
        result : neurokernel.plsel.BasePortMapper
            New port mapper instance.
        """

        assert isinstance(pm, cls)
        r = cls('')
        r.portmap = pm.portmap.copy()
        return r

    @property
    def index(self):
        """
        Port mapper index.
        """

        return self.portmap.index
    @index.setter
    def index(self, i):
        self.portmap.index = i

    def inds_to_ports(self, inds):
        """
        Convert list of integer indices to port identifiers.

        Examples
        --------
        >>> pm = BasePortMapper('/[a,b][0:2]')
        >>> print pm.inds_to_ports([0, 1])
        [('a', 0), ('a', 1)]

        Parameters
        ----------
        inds : array_like of int
            Integer indices of ports.

        Returns
        -------
        t : list of tuple
            Expanded port identifiers.
        """

        return self.portmap[self.portmap.isin(inds)].index.tolist()

    def ports_to_inds(self, selector):
        """
        Convert port selector to list of integer indices.

        Examples
        --------
        >>> pm = BasePortMapper('/[a,b][0:2]')
        >>> print pm.ports_to_inds('/b[0:2]')

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        inds : numpy.ndarray of int
            Integer indices of ports comprised by selector.
        """

        return self.sel.select(self.portmap,
                    selector).dropna().astype(np.int_).values

    def get_map(self, selector):
        """
        Retrieve integer indices associated with selector.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : numpy.ndarray
            Selected data.
        """

        return np.asarray(self.sel.select(self.portmap, selector).dropna())

    def set_map(self, selector, portmap):
        """
        Set mapped integer index associated with selector.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).
        portmap : sequence of int
            Integer indices to map to port identifiers.
        """

        self.portmap[self.sel.get_index(self.portmap, selector)] = portmap

    def equals(self, pm):
        """
        Check whether this mapper is equivalent to another mapper.

        Parameters
        ----------
        pm : neurokernel.plsel.BasePortMapper
            Mapper to compare to this mapper.

        Returns
        -------
        result : bool
             True if the specified port mapper contains the same port
             identifiers as this instance and maps them to the same integer
             values.

        Notes
        -----
        The port identifiers and maps in the specified port mapper need not be
        in the same order as this instance to be deemed equal.
        """

        assert isinstance(pm, BasePortMapper)
        pm0 = self.portmap.sort_values()
        pm1 = pm.portmap.sort_values()
        if np.array_equal(pm0.values, pm1.values) and \
           pm0.index.equals(pm1.index):
            return True
        else:
            return False

    def __len__(self):
        return self.portmap.size

    def __repr__(self):
        return 'Map:\n----\n'+self.portmap.__repr__()

class PortMapper(BasePortMapper):
    """
    Maps a numpy array to/from path-like port identifiers.

    Examples
    --------
    >>> data = np.array([1, 0, 3, 2, 5, 2])
    >>> pm = PortMapper('/d[0:5]', data)
    >>> print pm['/d[1]']
    array([0])
    >>> print pm['/d[2:4]']
    array([3, 2])

    Parameters
    ----------
    selector : str, unicode, or sequence
        Selector string (e.g., '/foo[0:2]') or sequence of token sequences
        (e.g., [['foo', (0, 2)]]) to map to `data`.
    data : numpy.ndarray
        1D data array to map to ports. If no data array is specified, port
        identifiers will still be mapped to their sequential indices but
        __getitem__() and __setitem__() will raise exceptions if invoked.
    portmap : sequence of int
        Integer indices to map to port identifiers. If no map is specified,
        it is assumed to be an array of consecutive integers from 0
        through one less than the number of ports.
    make_copy : bool
        If True, map a copy of the specified data array to the specified
        port identifiers.

    Attributes
    ----------
    data : numpy.ndarray
        Data that has been mapped to ports.
    dtype : numpy.dtype
        Type of mapped data.
    index : pandas.MultiIndex
        Index of port identifiers.
    portmap : pandas.Series
        Map of port identifiers to integer indices into `data`.

    Notes
    -----
    The selectors may not contain any '*' or '[:]' characters.
    """

    def _validate_data(self, data):
        """
        Check whether the mapper's ports are compatible with the specified port data array.
        """

        # A port mapper may contain or be assigned None as its data array:
        if data is None:
            return True
        try:
            # Cannot handle more than 1 dimension:
            assert np.ndim(data) <= 1

            # The integers in the port map must be valid indices into the
            # data array:
            # assert max(self.portmap) < len(data)

            # The port mapper may map identifiers to some portion of the data array:
            # assert len(self) <= len(data)
        except:
            return False
        else:
            return True

    def __init__(self, selector, data=None, portmap=None, make_copy=True):
        super(PortMapper, self).__init__(selector, portmap)

        self._data = None
        if data is None:
            self.data = None
        else:
            if np.ndim(data) == 0:
                self.data = np.full(len(self), data)
            else:
                if make_copy:
                    self.data = data.copy()
                else:
                    self.data = data

    @property
    def data(self):
        """
        Data associated with ports.
        """

        return self._data

    @data.setter
    def data(self, x):
        if self._validate_data(x):
            if x is None:
                self._data = None

            # Always store dimensionless values in a 1D array:
            elif np.ndim(x) == 0:
                self._data = np.array([x])
            else:
                if len(x):
                    self._data = np.asarray(x)
                else:
                    self._data = None
        else:
            raise ValueError('incompatible or invalid data array specified')

    def copy(self):
        """
        Return copy of this port mapper.

        Returns
        -------
        result : neurokernel.plsel.PortMapper
            Copy of port mapper instance.
        """

        c = self.__class__('')
        c.portmap = self.portmap.copy()
        c.data = self.data.copy()
        return c

    @classmethod
    def from_index(cls, idx, data, portmap=None):
        raise NotImplementedError

    @classmethod
    def from_pm(cls, pm):
        """
        Create a new port mapper instance given an existing instance.

        Parameters
        ----------
        result : neurokernel.plsel.PortMapper
            Existing port mapper instance.

        Returns
        -------
        result : neurokernel.plsel.PortMapper
            New port mapper instance.
        """

        assert isinstance(pm, cls)
        r = cls('')
        r.portmap = pm.portmap.copy()
        r.data = pm.data.copy()
        return r

    @property
    def dtype(self):
        """
        Port mapper data type.
        """

        return self.data.dtype
    @dtype.setter
    def dtype(self, d):
        self.data.dtype = d

    def get(self, selector):
        """
        Retrieve mapped data specified by given selector.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : numpy.ndarray
            Selected data.
        """

        if self.data is None:
            raise ValueError('port mapper contains no data')
        return self.data[np.asarray(self.sel.select(self.portmap, selector).dropna().values, dtype=np.int)]

    def get_by_inds(self, inds):
        """
        Retrieve mapped data specified by integer index.

        Parameters
        ----------
        inds : sequence of int
            Integer indices of data elements to return.

        Returns
        -------
        result : numpy.ndarray
            Selected data.
        """

        if self.data is None:
            raise ValueError('port mapper contains no data')
        return self.data[inds]

    def get_ports(self, f):
        """
        Select ports using a data selection function.

        Parameters
        ----------
        f : callable or sequence
            If callable, treat as elementwise selection function to apply to
            the mapped data array. If a sequence, treat as an index into the
            mapped data array.

        Returns
        -------
        s : list of tuple
            Expanded port identifiers selected by the specified function
            or boolean array.
        """

        assert callable(f) or (np.iterable(f) and len(f) == len(self.data))
        if callable(f):
            idx = self.portmap[f(self.data)].index
        else:
            idx = self.portmap[f].index
        return self.sel.index_to_selector(idx)

    def get_inds_nonzero(self):
        """
        Select indices of ports with nonzero data.

        Returns
        -------
        inds : numpy.ndarray
            Array of integer indices.
        """

        return np.nonzero(self.data)[0]

    def get_ports_nonzero(self):
        """
        Select ports with nonzero data.

        Returns
        -------
        s : list of tuple
            Expanded port identifiers whose corresponding data is nonzero.
        """
        return self.get_ports(lambda x: np.nonzero(x)[0])

    def get_ports_as_inds(self, f):
        """
        Select integer indices corresponding to ports in map.

        Examples
        --------
        >>> import numpy as np
        >>> pm = PortMapper(np.array([0, 1, 0, 1, 0]), '/a[0:5]')
        >>> pm.get_ports_as_inds(lambda x: np.asarray(x, dtype=np.bool))
        array([1, 3])

        Parameters
        ----------
        f : callable or sequence
            If callable, treat as elementwise selection function to apply to
            the mapped data array. If a sequence, treat as an index into the
            mapped data array.

        Returns
        -------
        inds : numpy.ndarray of int
            Integer indices of selected ports.
        """

        assert callable(f) or (np.iterable(f) and len(f) == len(self.data))
        if callable(f):
            v = self.portmap[f(self.data)].values
        else:
            v = self.portmap[f].values
        return v

    def set(self, selector, data):
        """
        Set mapped data specified by given selector.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).
        data : numpy.ndarray
            Array of data to save.
        """

        # sel.select will return a Series with nan for selector [()], hence dropna
        # is necessary here
        if self.data is None:
            self.data = data
        else:
            self.data[np.asarray(self.sel.select(self.portmap, selector).dropna().values, dtype=np.int)] = data

    def set_by_inds(self, inds, data):
        """
        Set mapped data by integer indices.

        Parameters
        ----------
        inds : sequence of int
            Integer indices of data elements to update.
        data : numpy.ndarray
            Data to assign.
        """

        self.data[inds] = data

    __getitem__ = get
    __setitem__ = set

    def equals(self, other):
        """
        Check whether this mapper is equivalent to another mapper.

        Parameters
        ----------
        other : neurokernel.plsel.PortMapper
            Mapper to compare to this mapper.

        Returns
        -------
        result : bool
            True if the mappers map the same selectors to the same integer
            indices and data.

        Notes
        -----
        Mappers containing the same rows in different orders are not
        regarded as equivalent.
        """

        assert isinstance(other, self.__class__)
        return self.portmap.equals(other.portmap) and (self.data == other.data).all()

    def __repr__(self):
        return 'Map:\n----\n'+self.portmap.__repr__()+'\n\ndata:\n'+self.data.__repr__()
