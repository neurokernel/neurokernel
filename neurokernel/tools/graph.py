#!/usr/bin/env python

"""
Graph/connectivity manipulation and visualization tools
"""

import itertools
import os
import re
import tempfile

import matplotlib.pyplot as plt
import networkx as nx

from .. import base

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

def conn_to_bipartite(c):
    """
    Convert a connectivity object into a bipartite NetworkX directed multigraph.

    Parameters
    ----------
    c : base.BaseConnectivity
        Connectivity object.
    
    """

    if not isinstance(c, base.BaseConnectivity):
        raise ValueError('invalid connectivity object')
    
    g = nx.MultiDiGraph()
    src_nodes = ['src:%i' % i for i in xrange(c.N_src)]
    dest_nodes = ['dest:%i' % i for i in xrange(c.N_dest)]
    g.add_nodes_from(src_nodes)
    g.add_nodes_from(dest_nodes)

    # First process connection keys:
    for key in [k for k in c._data.keys() if re.match('.*\/conn$', k)]:
        syn, dir, name = key.split('/')
        syn = int(syn)
        for src, dest in itertools.product(xrange(c.N_src), xrange(c.N_dest)):
            if c[src, dest, syn, dir, name] == 1:
                if dir == '+':
                    g.add_edge('src:%i' % src, 'dest:%i' % dest, syn)
                elif dir == '-':
                    g.add_edge('dest:%i' % dest, 'src:%i' % src, syn)
                else:
                    raise ValueError('unrecognized direction')

    # Next, process attributes:
    for key in [k for k in c._data.keys() if not re.match('.*\/conn$', k)]:
        syn, dir, name = key.split('/')
        syn = int(syn)
        for src, dest in itertools.product(xrange(c.N_src), xrange(c.N_dest)):

            # Parameters are only defined for node pairs for which there exists
            # a connection:
            if c[src, dest, syn, dir] == 1:
                print src,dest,syn,dir,name,c[src, dest, syn, dir, name]
                if dir == '+':
                    try:
                        g['src:%i' % src]['dest:%i' % dest][syn][name] = \
                            c[src, dest, syn, dir, name]                
                    except:
                        pass
                elif dir == '-':
                    try:
                        g['dest:%i' % dest]['src:%i' % src][syn][name] = \
                            c[src, dest, syn, dir, name]
                    except:
                        pass
                else:
                    raise ValueError('unrecognized direction')
            
    return g

def bipartite_to_conn(g):
    """
    Convert a bipartite NetworkX directed multigraph.

    Parameters
    ----------
    g : networkx.MultiDiGraph
        Directed multigraph instance.
    
    Notes
    -----
    Assumes that all nodes are labels 'src:X' or 'dest:X'.
    
    """

    if not isinstance(g, nx.MultiDiGraph):
        raise ValueError('invalid graph object')
    if not nx.is_bipartite(g):
        raise ValueError('graph must be bipartite')

    # Categorize nodes and determine number of nodes to support:
    N_src = 0
    N_dest = 0
    src_nodes = []
    dest_nodes = []
    for label in g.nodes():
        try:
            n = int(re.search('src:(\d+)', label).group(1))
        except:
            pass
        else:
            src_nodes.append(n)
            if n+1 > N_src:
                N_src = n+1
            continue
        
        try:
            n = int(re.search('dest:(\d+)', label).group(1))
        except:
            raise ValueError('unrecognized node label')
        else:
            dest_nodes.append(n)
            if n+1 > N_dest:
                N_dest = n+1
            continue
            
    # Find maximum number of edges between any two nodes:
    N_mult = [len(g[u][v]) for u,v in set(g.edges())]

    # Create empty connectivity structure:
    c = base.BaseConnectivity(N_src, N_dest, N_mult)

    # Populate:
    for i, j in itertools.product(src_nodes, dest_nodes):

        # Check connections from src to dest nodes and from
        # dest to src nodes:
        for src, dest, dir in [('src:%i' % i, 'dest:%i' % j, '+'),
                               ('dest:%i' % j, 'src:%i' % i, '-')]:

            try:
                edge = g[src][dest]
            except:
                pass
            else:

                # Loop through all synapses:
                for syn in edge.keys():
                    c[i,j,syn,dir] = 1

                    # Loop through all synaptic parameters:
                    for param in edge[syn].keys():
                        c[i,j,syn,dir,param] = edge[syn][param]
    return c                        
        
