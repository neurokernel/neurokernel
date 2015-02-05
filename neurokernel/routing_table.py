#!/usr/bin/env python

"""
Routing table class.
"""

import numpy as np
import networkx as nx
import pandas as pd

class RoutingTable(object):
    """
    Routing table class.

    Simple class that stores pairs of strings that can signify
    one-hop routes between entities in a graph. Assigning a value to a
    pair that isn't in the class instance will result in the pair and value
    being added to the class instance.

    Parameters
    ----------
    g : networkx.DiGraph
        Directed graph that describes the routes between entities.

    Attributes
    ----------
    connections : list
        List of directed connections between identifiers.
    ids : list
        Identifiers currently in routing table.

    Methods
    -------
    copy()
        Return a copy of the routing table.
    dest_ids(src_id)
        Destination identifiers connected to the specified source identifier.
    ids()
        IDs currently in routing
    src_ids(dest_id)
        Source identifiers connected to the specified destination identifier.
    to_df()
        Return a pandas DataFrame listing all of the connections.
    """

    def __init__(self, g=None):
        if g is None:
            self.data = nx.DiGraph()
        else:
            assert type(g) == nx.DiGraph
            self.data = g

    def __setitem__(self, key, value):
        assert type(key) == tuple
        assert len(key) >= 2

        if not self.data.has_node(key[0]):
            self.data.add_node(key[0])
        if not self.data.has_node(key[1]):
            self.data.add_node(key[1])
        if len(key) > 2:
            if np.isscalar(value):
                data = {k: value for k in key[2:]}
            elif type(value) == dict:
                data = value
            elif np.iterable(value) and len(value) <= len(key[2:]):
                data = {k: v for k, v in zip(key[2:], value)}
            else:
                raise ValueError('cannot assign specified value')
        else:
            if np.isscalar(value):
                data = {'conn': value}
            elif type(value) == dict:
                data = value
            else:
                raise ValueError('cannot assign specified value')
        self.data.add_edge(key[0], key[1], data)

    def __getitem__(self, key):
        assert type(key) == tuple
        assert len(key) >= 2
        if len(key) > 2:
            result = [self.data.edge[key[0]][key[1]][k] for k in key[2:]]
        else:
            result = [self.data.edge[key[0]][key[1]][k] for k in \
                      self.data.edge[key[0]][key[1]].keys()]
        if len(result) == 1:
            return result[0]
        else:
            return result
            return self.data.edge[key[0]][key[1]]

    def __copy__(self):
        r = self.__class__()
        r.data = self.data.copy()

    copy = __copy__

    @property
    def ids(self):
        """
        Identifiers currently in routing table.
        """

        return self.data.nodes()

    @property
    def connections(self):
        """
        List of directed connections between identifiers.
        """

        return self.data.edges()

    def src_ids(self, dest_id):
        """
        Source identifiers connected to the specified destination identifier.
        """

        return self.data.predecessors(dest_id)

    def dest_ids(self, src_id):
        """
        Destination identifiers connected to the specified source identifier.
        """

        return self.data.successors(src_id)

    def to_df(self):
        """
        Return a pandas DataFrame listing all of the connections.
        """

        tuples = []
        data = []
        for t in self.data.edges_iter(data=True):
            tuples.append(t[0:2])
            data.append(t[2])
        if tuples:
            idx = pd.MultiIndex.from_tuples(tuples)
            idx.names = ['from', 'to']
        else:
            idx = pd.MultiIndex(levels=[[], []],
                                labels=[[], []],
                                names=['from', 'to'])
        df = pd.DataFrame.from_dict(data)
        df.index = idx
        return df

    def __repr__(self):
        return self.to_df().__repr__()
