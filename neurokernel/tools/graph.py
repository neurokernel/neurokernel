#!/usr/bin/env python

"""
Graph/connectivity manipulation and visualization tools
"""

import glob
import itertools
import os.path
import re

import networkx as nx

# Work around bug in networkx < 1.9 that causes networkx to choke on GEXF 
# files with boolean attributes that contain the strings 'True' or 'False'
# (bug already observed in https://github.com/networkx/networkx/pull/971)
nx.readwrite.gexf.GEXF.convert_bool['false'] = False
nx.readwrite.gexf.GEXF.convert_bool['False'] = False
nx.readwrite.gexf.GEXF.convert_bool['true'] = True
nx.readwrite.gexf.GEXF.convert_bool['True'] = True

import pandas

from .. import base
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

    # Extract the node/edge data:
    try:
        node_data = {int(k): v for k, v in g.node.iteritems()}
    except:
        node_data = {k: v for k, v in g.node.iteritems()}

    try:
        if not isinstance(g, nx.MultiGraph):
            edge_data = {(int(k1), int(k2)):v for k1 in g.edge.keys() \
                             for k2, v in g.edge[k1].iteritems()}

        else:

            # Include the edge number in the index for multigraphs:
            edge_data = {(int(k1), int(k2), int(m)):v for k1 in g.edge.keys() \
                             for k2 in g.edge[k1].keys() \
                             for m, v in g.edge[k1][k2].iteritems()}
    except:
        if not isinstance(g, nx.MultiGraph):
            edge_data = {(k1, k2):v for k1 in g.edge.keys() \
                             for k2, v in g.edge[k1].iteritems()}

        else:

            # Include the edge number in the index for multigraphs:
            edge_data = {(k1, k2, m):v for k1 in g.edge.keys() \
                             for k2 in g.edge[k1].keys() \
                             for m, v in g.edge[k1][k2].iteritems()}
        
    # Construct DataFrame instances:
    df_node = pandas.DataFrame.from_dict(node_data).T
    df_edge = pandas.DataFrame.from_dict(edge_data).T

    # Convert edge index to MultiIndex to facilitate access using edge endpoints
    # and number (for multigraphs):
    df_edge.index = pandas.MultiIndex.from_tuples(df_edge.index)

    return df_node, df_edge

