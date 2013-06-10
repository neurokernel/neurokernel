#!/usr/bin/env python

import itertools
import os
import tempfile

import core
import matplotlib.pyplot as plt
import networkx as nx

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
    Convert a Connectivity object into a bipartite NetworkX multigraph.
    """

    if not isinstance(c, core.BaseConnectivity):
        raise ValueError('invalid connectivity object')
    
    g = nx.MultiDiGraph()
    src_nodes = ['src_%i' % i for i in xrange(c.N_src)]
    dest_nodes = ['dest_%i' % i for i in xrange(c.N_dest)]
    g.add_nodes_from(src_nodes)
    g.add_nodes_from(dest_nodes)

    for key in c._data.keys():
        syn, dir, name = key.split('/')
        syn = int(syn)
        if name == 'conn':
            if dir == '+':
                for src, dest in itertools.product(xrange(c.N_src), xrange(c.N_dest)):
                    if c[src, dest, syn, dir, name] == 1:
                        g.add_edge('src_%i' % src, 'dest_%i' % dest)
            elif dir == '-':
                for src, dest in itertools.product(xrange(c.N_src), xrange(c.N_dest)):
                    if c[src, dest, syn, dir, name] == 1:
                        g.add_edge('dest_%i' % dest, 'src_%i' % src)
            else:
                raise ValueError('invalid direction')
    return g
