#!/usr/bin/env python

"""
Graph/connectivity manipulation and visualization tools
"""

import glob
import itertools
import os.path
import re
from future.utils import iteritems

import networkx as nx

# Work around bug in networkx < 1.9 that causes networkx to choke on GEXF
# files with boolean attributes that contain the strings 'True' or 'False'
# (bug already observed in https://github.com/networkx/networkx/pull/971)
nx.readwrite.gexf.GEXF.convert_bool['false'] = False
nx.readwrite.gexf.GEXF.convert_bool['False'] = False
nx.readwrite.gexf.GEXF.convert_bool['true'] = True
nx.readwrite.gexf.GEXF.convert_bool['True'] = True

import pandas

from .. import core

def graph_to_df(g):
    """
    Convert a directed multigraph into pandas DataFrames.

    Constructs two pandas DataFrame instances that respectively contain the
    specified graph's node and edge attributes.

    Parameters
    ----------
    g : networkx.Graph
        Graph instance.

    Returns
    -------
    df_node : pandas.DataFrame
        Node attributes; the index corresponds to the node identifiers.
    df_edge : pandas.DataFrame
        Edge attributes; the index is a MultiIndex that permits
        access to the edges by origin node, destination node, and edge number
        (i.e., for multigraphs in which nodes connected by more than one
        edge with the same direction).

    """

    nx_major_version = int(nx.__version__.split('.')[0])

    # Extract the node/edge data:
    if nx_major_version < 2:
        node_data = {k: v for k, v in iteritems(g.node)}
        if not isinstance(g, nx.MultiGraph):
            edge_data = {(k1, k2): v for k1 in g.edge \
                                        for k2, v in iteritems(g.edge[k1])}
        else:
            edge_data = {(k1, k2, m): v for k1 in g.edge \
                             for k2 in g.edge[k1] \
                             for m, v in iteritems(g.edge[k1][k2])}
    else:
        node_data = {k: v for k, v in iteritems(g.nodes)}
        edge_data = {k1: k2 for k1, k2 in iteritems(g.edges)}

    # Construct DataFrame instances:
    df_node = pandas.DataFrame.from_dict(node_data).T
    df_edge = pandas.DataFrame.from_dict(edge_data).T

    # Convert edge index to MultiIndex to facilitate access using edge endpoints
    # and number (for multigraphs):
    df_edge.index = pandas.MultiIndex.from_tuples(df_edge.index)

    return df_node, df_edge
