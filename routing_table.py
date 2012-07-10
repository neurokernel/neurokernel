import numpy as np
import la

class RoutingTable(object):
    """
    Routing table.

    Parameters
    ----------
    t : la.larry
       Labeled array to use when initializing table.

    Notes
    -----
    Inserting rows or columns of values is not currently supported.

    The initial array must be 2D and possess the same list labels for
    both of its dimensions.

    """

    def __init__(self, t=None):
        if t is None:
            self._data = None
        else:
            try:
                type(t) == la.larry
                t.label[0] == t.label
            except:
                raise ValueError('invalid initial array')
            else:
                self._data = t.copy()

    def __setitem__(self, key, value):
        if type(key) == slice:
            raise ValueError('assignment by slice not supported')
        if len(key) != 2:
            raise KeyError('invalid key')
        row, col = key
        if not self._data:
            label = list(set(key))
            self._data = la.larry(np.zeros(2*(len(label),), type(value)),
                                  [label, label])
        else:

            # If either the row or column identifier isn't in the
            # current list of labels, add it:
            for k in key:

                # Create new row:
                if k not in self._data.label[0]:
                    self._data = self._data.merge(la.larry([[0]*len(self._data.label[1])],
                                                           [[k], self._data.label[1]]))

                # Create new column:
                if k not in self._data.label[1]:
                    self._data = self._data.merge(la.larry([[0]]*len(self._data.label[0]),
                                                           [self._data.label[0], [k]]))
        self._data.set([row, col], int(value))

    def __getitem__(self, key):

        # Index values referring to labels must be wrapped with lists:
        reformat_slice = lambda s: slice(s.start if s.start is None else [s.start],
                                         s.stop if s.stop is None else [s.stop],
                                         s.step)
        if type(key) == tuple:
            key = tuple([reformat_slice(k) if type(k) == slice else [k] for k in key])
        elif type(key) == slice:
            key = reformat_slice(key)
        else:
            key = [key]
        return self._data.lix[key]

    def __copy__(self):
        return RoutingTable(self._data)

    copy = __copy__

    @property
    def shape(self):
        """
        Shape of table.
        """

        return self._data.shape

    @property
    def ids(self):
        """
        IDs currently in routing table.
        """

        if self._data is None:
            return []
        else:
            return self._data.label[0]

    def row_ids(self, col_id):
        """
        Row IDs connected to a column ID.
        """

        return [self[:, col_id].label[0][i] for i, e in \
                enumerate(self[:, col_id]) if e != 0]

    def all_row_ids(self):
        """
        All row IDs connected to column IDs.
        """

        return [self._data.label[0][i] for i, e in \
                enumerate(np.sum(self._data.x, 1, np.bool)) if e]
                
    def col_ids(self, row_id):
        """
        Column IDs connected to a row ID.
        """

        return [self[row_id, :].label[0][i] for i, e in \
                enumerate(self[row_id, :]) if e != 0]

    def all_col_ids(self):
        """
        All column IDs connected to row IDs.
        """

        return [self._data.label[0][i] for i, e in \
                enumerate(np.sum(self._data.x, 0, np.bool)) if e]

    def __repr__(self):
        if self._data is None:
            return 'empty'
        else:
            t = 'ids:   ' + repr(self.ids) + '\n' + \
              self._data.getx().__repr__()
            return t

if __name__ == '__main__':
    t = RoutingTable()
    t['a', 'b'] = 1
    t['b', 'c'] = 1
    print t

