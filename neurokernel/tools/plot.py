#!/usr/bin/env python

"""
Plotting tools
"""

import os
import tempfile

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
