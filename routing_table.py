import pandas

class DynamicTable(object):
    """
    Dynamically sized table.
    """

    def __init__(self, *args, **kwargs):
        self._data = pandas.DataFrame(*args, **kwargs)

    def __getitem__(self, key):
        """
        Retrieve a value from the table.

        Note
        ----
        The first index in the key specifies the column.
        """

        if len(key) != 2:
            raise KeyError('invalid key')

        col, row = key
        return self._data.__getitem__(col).__getitem__(row)

    def __setitem__(self, key, value):
        """
        Set the specified entry in the table.

        Notes
        -----
        The first index in the key specifies the column.

        If the specified row or column identifiers do not exist, the
        table is expanded to include rows or columns with those
        identifiers.

        """

        if len(key) != 2:
            raise KeyError('invalid key')
        col, row = key
        if row not in self.data.index:
            new_row = pandas.DataFrame(index=[row],                                       
                                       columns=self._data.columns)
            self._data = pandas.concat([self._data, new_row])
        if col not in self._data.columns:
            new_col = pandas.DataFrame(index=self._data.index,              
                                       columns=[col])
            self._data = pandas.concat([self._data, new_col], axis=1)
        self._data[col][row] = value

    @property
    def table(self):
        """
        Return a view of the current table.
        """

        return self._data

    def __repr__(self):
        return self._data.__repr__()

class RoutingTable(DynamicTable):
    """
    Routing table.
    """

    def __init__(self):
        DynamicTable.__init__(self)

    def __setitem__(self, key, value):
        """
        Set the specified entry in the table.

        Notes
        -----
        The first index in the key specifies the column.

        If the specified row or column identifiers do not exist, the
        table is expanded to include rows or columns with those
        identifiers.

        """

        if len(key) != 2:
            raise KeyError('invalid key')
        col, row = key

        # Since the routing table must describe routes between all
        # recognized entities, adding a hitherto unrecognized row or
        # column identifier must cause that identifier to be added to
        # both the list of rows and columns:
        Nc = len(self._data.columns)
        Nr = len(self._data.index)
        for k in (col, row):
            if k not in self._data.index:
                new_row = pandas.DataFrame(index=[k],
                                           columns=self._data.columns)
                self._data = pandas.concat([self._data, new_row])
            if k not in self._data.columns:
                new_col = pandas.DataFrame(index=self._data.index,                      
                                           columns=[k])
                self._data = pandas.concat([self._data, new_col], axis=1)
        self._data[col][row] = value

    @property
    def ids(self):
        """
        Identifiers of rows and columns in the routing table.
        """

        return self._data.index.tolist()
