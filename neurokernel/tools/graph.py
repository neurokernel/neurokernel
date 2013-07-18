#!/usr/bin/env python

"""
Graph/connectivity manipulation and visualization tools
"""

import glob
import itertools
import os.path
import re
import tempfile

import matplotlib.pyplot as plt
import networkx as nx

from .. import base
from .. import core

def imdisp(f):
    """
    Display the specified image file using matplotlib.
    """
    
    im = plt.imread(f)
    plt.imshow(im)
    plt.axis('off')
    plt.draw()
    return im

def show_pydot(g):
    """
    Display a networkx graph using pydot.
    """
    
    fd = tempfile.NamedTemporaryFile()
    fd.close()
    p = nx.to_pydot(g)
    p.write_jpg(fd.name)
    imdisp(fd.name)
    os.remove(fd.name)

def show_pygraphviz(g, prog='dot', graph_attr={}, node_attr={}, edge_attr={}):
    """
    Display a networkx graph using pygraphviz.

    Parameters
    ----------
    prog : str
        Executable for generating the image.
    graph_attr : dict
        Global graph display attributes.
    node_attr : dict
        Global node display attributes.
    edge_attr : dict
        Global edge display attributes.
    
    """
    
    fd = tempfile.NamedTemporaryFile(suffix='.jpg')
    fd.close()
    p = nx.to_agraph(g)
    p.graph_attr.update(graph_attr)
    p.node_attr.update(node_attr)
    p.edge_attr.update(edge_attr)
    p.draw(fd.name, prog=prog)
    imdisp(fd.name)
    os.remove(fd.name)

def conn_to_graph(c):
    """
    Convert a connectivity object into a bipartite NetworkX directed multigraph.

    Parameters
    ----------
    c : {base.BaseConnectivity, core.Connectivity}
        Connectivity object.

    Examples
    --------
    >>> c = base.BaseConnectivity(2, 3)
    >>> c['A', 0, 'B', 1] = 1
    >>> c['A', 1, 'B', 2] = 1
    >>> g = conn_to_graph(c)
    >>> g.nodes()
    ['A:0', 'A:1', 'B:2', 'B:1', 'B:0']
    >>> g.edges()
    [('A:0', 'B:1'), ('A:1', 'B:2')]
    
    """

    if not isinstance(c, base.BaseConnectivity):
        raise ValueError('invalid connectivity object')
    
    g = nx.MultiDiGraph()
    A_nodes = [c.A_id+':'+str(i) for i in xrange(c.N_A)]
    B_nodes = [c.B_id+':'+str(i) for i in xrange(c.N_B)]
    g.add_nodes_from(A_nodes)
    g.add_nodes_from(B_nodes)

    # Set the neuron_type attribute of the nodes if
    # the specified object is a Connectivity instance:
    if isinstance(c, core.Connectivity):
        for i in xrange(c.N_A_gpot):
            g.node[c.A_id+':'+\
                str(c.idx_translate[c.A_id]['gpot', i])]['neuron_type'] = 'gpot'
        for i in xrange(c.N_B_gpot):
            g.node[c.B_id+':'+\
                str(c.idx_translate[c.B_id]['gpot', i])]['neuron_type'] = 'gpot'
        for i in xrange(c.N_A_spike):
            g.node[c.A_id+':'+\
                str(c.idx_translate[c.A_id]['spike', i])]['neuron_type'] = 'spike'
        for i in xrange(c.N_B_spike):
            g.node[c.B_id+':'+\
                str(c.idx_translate[c.B_id]['spike', i])]['neuron_type'] = 'spike'
            
    # Create the edges of the graph:
    for key in [k for k in c._data.keys() if re.match('.*\/conn$', k)]:
        A_id, B_id, conn, name = key.split('/')
        conn = int(conn)
        for i, j in itertools.product(xrange(c.N(A_id)), xrange(c.N(B_id))):
            if isinstance(c, core.Connectivity):
                if c[A_id, 'all', i, B_id, 'all', j, conn, name] == 1:
                    g.add_edge(A_id+':'+str(i), B_id+':'+str(j))                
            else:
                if c[A_id, i, B_id, j, conn, name] == 1:
                    g.add_edge(A_id+':'+str(i), B_id+':'+str(j))


    # Next, set the attributes on each edge: 
    for key in [k for k in c._data.keys() if not re.match('.*\/conn$', k)]:
        A_id, B_id, conn, name = key.split('/')
        conn = int(conn)
        for i, j in itertools.product(xrange(c.N(A_id)), xrange(c.N(B_id))):
            
            # Parameters are only defined for node pairs for which there exists
            # a connection:
            if isinstance(c, core.Connectivity):
                if c[A_id, 'all', i, B_id, 'all', j, conn, name]:                
                    g[A_id+':'+str(i)][B_id+':'+str(j)][conn][name] = \
                        c[A_id, 'all', i, B_id, 'all', j, conn, name]  
            else:
                if c[A_id, i, B_id, j, conn, name]:                
                    g[A_id+':'+str(i)][B_id+':'+str(j)][conn][name] = \
                        c[A_id, i, B_id, j, conn, name]            
    return g

def graph_to_conn(g, conn_type=core.Connectivity):
    """
    Convert a bipartite NetworkX directed multigraph to a connectivity object.

    Parameters
    ----------
    g : networkx.MultiDiGraph
        Directed multigraph instance.
    conn_type : {base.BaseConnectivity, core.Connectivity}
        Type of output to generate.

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.MultiDiGraph()
    >>> g.add_nodes_from(['A:0', 'A:1', 'B:2', 'B:1', 'B:0'])
    >>> g.add_edges_from([('A:0', 'B:1'), ('A:1', 'B:2')])
    >>> c = graph_to_conn(g, base.BaseConnectivity)
    >>> c['A', :, 'B', :]
    array([[0, 1, 0],
           [0, 0, 1]])
        
    Notes
    -----
    Assumes that `g` is bipartite and all of its nodes are labeled 'A:X' or
    'B:X', where 'A' and 'B' are the names of the connected modules.

    When loading a graph from a GEXF file via networkx.read_gexf(),
    the relabel parameter should be set to True to prevent the actual labels in
    the file from being ignored.
    """

    if not isinstance(g, nx.MultiDiGraph):
        raise ValueError('invalid graph object')
    if not nx.is_bipartite(g):
        raise ValueError('graph must be bipartite')
    if conn_type not in [base.BaseConnectivity, core.Connectivity]:
        raise ValueError('invalid connectivity type')
    
    # Categorize nodes and determine number of nodes to support:
    A_nodes = []
    B_nodes = []

    node_dict = {}
    for label in g.nodes():
        try:
            id, n = re.search('(.+):(.+)', label).groups()
        except:
            raise ValueError('incorrectly formatted node label: %s' % label)
        if not node_dict.has_key(id):
            node_dict[id] = set()
        node_dict[id].add(int(n))
    
    # Graph must be bipartite:
    if len(node_dict.keys()) != 2:
        raise ValueError('incorrectly formatted graph')
    A_id, B_id = node_dict.keys()
    N_A = len(node_dict[A_id])
    N_B = len(node_dict[B_id])

    # Nodes must be consecutively numbered from 0 to N:
    if set(range(max(node_dict[A_id])+1)) != node_dict[A_id] or \
        set(range(max(node_dict[B_id])+1)) != node_dict[B_id]:
        raise ValueError('nodes must be numbered consecutively from 0..N')

    # If a Connectivity object must be created, count how many graded potential
    # and spiking neurons are comprised by the graph:
    if conn_type == core.Connectivity:
        N_A_gpot = 0
        N_A_spike = 0
        N_B_gpot = 0
        N_B_spike = 0
        for n in node_dict[A_id]:
            if g.node[A_id+':'+str(n)]['neuron_type'] == 'gpot':
                N_A_gpot += 1
            elif g.node[A_id+':'+str(n)]['neuron_type'] == 'spike':
                N_A_spike += 1
            else:
                raise ValueError('invalid neuron type')
        for n in node_dict[B_id]:
            if g.node[B_id+':'+str(n)]['neuron_type'] == 'gpot':
                N_B_gpot += 1
            elif g.node[B_id+':'+str(n)]['neuron_type'] == 'spike':
                N_B_spike += 1
            else:
                raise ValueError('invalid neuron type')
        
    # Find maximum number of edges between any two nodes:
    N_mult = max([len(g[u][v]) for u,v in set(g.edges())])

    # Create empty connectivity structure:
    if conn_type == base.BaseConnectivity:
        c = base.BaseConnectivity(N_A, N_B, N_mult, A_id, B_id)
    else:
        c = core.Connectivity(N_A_gpot, N_A_spike, N_B_gpot, N_B_spike,
                              N_mult, A_id, B_id)
    
    # Set the parameters in the connectivity object:
    for edge in g.edges():
        A_id, i = edge[0].split(':')
        i = int(i)
        B_id, j = edge[1].split(':')
        j = int(j)
        edge_dict = g[edge[0]][edge[1]]
        for conn, k in enumerate(edge_dict.keys()):
            if conn_type == base.BaseConnectivity:
                idx_tuple = (A_id, i, B_id, j, conn)
            else:
                idx_tuple = (A_id, 'all', i, B_id, 'all', j, conn)
            c[idx_tuple] = 1

            for param in edge_dict[k].keys():

                # The ID loaded by networkx.read_gexf() is always a string, but
                # should be interpreted as an integer:
                if param == 'id':
                    c[idx_tuple+(param,)] = int(edge_dict[k][param])
                else:
                    c[idx_tuple+(param,)] = float(edge_dict[k][param])                        
                                    
    return c                        
        
def load_conn_all(data_dir, conn_type=core.Connectivity):
    """
    Load all emulation inter-module connectivity data from a specified directory.

    Searches for GEXF files describing modules and connectivity objects,
    verifies that the connectivity objects actually do refer to the modules,
    and verifies that the connectivity objects and the modules that they
    connect are compatible.

    Parameters
    ----------
    data_dir : str
        Directory containing emulation files
    conn_type : {base.BaseConnectivity, core.Connectivity}
        Type of connectivity object to create.
        
    Returns
    -------
    mod_dict : dict
        Dictionary mapping module file base names to
        NetworkX MultiDiGraph instances.
    conn_dict : dict
        Dictionary mapping connectivity file base names to
        generated connectivity object instances.
    """

    gexf_list = glob.glob(os.path.join(data_dir, '*.gexf'))
    mod_dict = {}
    conn_dict = {}
    for file_name in gexf_list:

        # Try to load inter-module connectivity data from a GEXF file; if the
        # load fails, the file is assumed to contain module information:
        g = nx.MultiDiGraph(nx.read_gexf(file_name, relabel=True))
        try:
            conn = graph.graph_to_conn(g)
        except:
            mod_dict[os.path.splitext(file_name)][0] = g
        else:
            conn_dict[os.path.splitext(file_name)][0] = conn

    # Check whether all of the loaded connectivity objects refer to the
    # specified module files
    for conn_name in conn_dict.keys():
        conn = conn_dict[conn_name]
        if conn.A_id not in mod_dict.keys():
            raise ValueError('module %s in connectivity object %s not found' % \
                             (conn.A_id, conn_name))
        if conn.B_id not in mod_dict.keys():
            raise ValueError('module %s in connectivity object %s not found' % \
                             (conn.B_id, conn_name))
    
    return mod_dict, conn_dict
    
