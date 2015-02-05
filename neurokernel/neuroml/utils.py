#!/usr/bin/env python

"""
Neurokernel NeuroML input/output routines.
"""

import networkx as nx

from nml import parse, parseString, NeurokernelDoc
from nml import Module, Interface, Pattern, PatternConnection, Port
from nml import MLNeuron, LifNeuron, AlSynapse, PGGSynapse

def write(nk_doc, file, root_name='nk'):
    """
    Write a Neurokernel NeuroML document to an XML file.

    Parameters
    ----------
    nk_doc : neurokernel.neuroml.NeurokernelDoc
        Neurokernel NeuroML document root object.
    file : str or file
        Output file name or handle.
    root_name : str
        Document root name.
    """

    assert isinstance(nk_doc, NeurokernelDoc)

    ns_def = 'xmlns="http://www.neuroml.org/schema/neuroml2"\n'
    ns_def += '    xmlns:xs="http://www.w3.org/2001/XMLSchema"\n'
    ns_def += '    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
    ns_def += '    xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2'
    ns_def += ' https://raw.github.com/neurokernel/neurokernel/development/neuroml/neurokernel.xsd"'

    # Set name_ param to ensure root element named correctly due to generateDS
    # limitation:
    try:
        nk_doc.export(file, 0, name_=root_name,
                      namespacedef_=ns_def) 
    except:
        try:
            with open(file, 'w') as f:
                nk_doc.export(f, 0, name_=root_name,
                              namespacedef_=ns_def) 
        except:
            raise RuntimeError('cannot write file')

def load(file):
    """
    Load a Neurokernel NeuroML document.

    Parameters
    ----------
    file : str or file
        Input file name or handle.

    Returns
    -------
    nk_doc : neurokernel.neuroml.NeurokernelDoc
        Neurokernel NeuroML document root.
    """

    try:        
        nk_doc = parse(file)
    except:
        try:
            nk_doc = parseString(file.read())
        except:
            raise RuntimeError('cannot load file')

    return nk_doc

def nml_pattern_to_graph(pattern):
    """
    Convert a pattern expressed in Neurokernel NeuroML into a NetworkX graph.

    Parameters
    ----------
    pattern : neurokernel.neuroml.Pattern
        Pattern instance.

    Returns
    -------
    g : networkx.DiGraph
        Directed graph containing the pattern's ports (nodes) and connections
        (edges).
    """

    assert isinstance(pattern, Pattern)

    g = nx.Diraph()
    for p in pattern.interface.ports:
        g.add_node(p.identifier)
        g.node[p.identifier] = {
            'interface': int(p.interface),
            'io': p.io,
            'type': p.type
            }

    for c in pattern.connections:
        g.add_edge(c.from_, c.to, type='directed')

    return g

def graph_to_nml_pattern(g, id):
    """
    Convert a pattern expressed as a NetworkX graph into Neurokernel NeuroML.

    Parameters
    ----------
    g : networkx.DiGraph
        Directed graph containing the pattern's ports (nodes) and connections
        (edges).
    id : str
        Pattern identifier.

    Returns
    -------
    pattern : neurokernel.neuroml.Pattern
        pattern instance.
    """

    pattern = Pattern(id=id)
    interface = Interface()
    for p in g.nodes():
        attr_dict = g.node[p]
        port = Port(identifier=attr_dict['identifier'],
                    interface=attr_dict['interface'],
                    io=attr_dict['io'],
                    type=attr_dict['type'])
        interface.ports.append(port)

    pattern.interface = interface
    for c in g.edges():
        connection = PatternConnection(from_=c[0], to=c[1])
        pattern.connections.append(connection)

    return pattern

def nml_module_to_graph(module):
    """
    Convert a module expressed in Neurokernel NeuroML into NetworkX graphs.

    Parameters
    ----------
    module : neurokernel.neuroml.Module
        Module instance.

    Returns
    -------
    g : networkx.DiGraph
        Directed graph containing a module's neurons and synapses
    i : networkx.Graph
        Undirected graph containing a module's interface ports.
    """

    assert isinstance(module, Module)

    # Ensure that none of the neurons or synapses have duplicate IDs:
    assert len(set([n.id for n in module.ml_neurons])) == \
        len(module.ml_neurons)
    assert len(set([n.id for n in module.lif_neurons])) == \
        len(module.lif_neurons)
    assert len(set([s.id for s in module.al_synapses])) == \
        len(module.al_synapses)
    assert len(set([s.id for s in module.pgg_synapses])) == \
        len(module.pgg_synapses)

    g = nx.DiGraph()    
    for n in module.ml_neurons:
        g.add_node(n.id)
        g.node[n.id] = {
            'model': 'MorrisLecar',
            'name': n.id,
            'extern': True if n.extern == 'true' else False,
            'public': True if n.public == 'true' else False,
            'spiking': True if n.spiking == 'true' else False,
            'V1': float(n.V1),
            'V2': float(n.V2),
            'V3': float(n.V3),
            'V4': float(n.V4),
            'phi': float(n.phi),
            'offset': float(n.offset),
            'initV': float(n.init_v),
            'initn': float(n.initn)
        }
    for n in module.lif_neurons:
        g.add_node(n.id)
        g.node[n.id] = {
            'model': 'LeakyIAF',
            'name': n.id,
            'extern': True if n.extern == 'true' else False,
            'public': True if n.public == 'true' else False,
            'spiking': True if n.spiking == 'true' else False,
            'V': float(n.V),
            'Vr': float(n.Vr),
            'Vt': float(n.Vt),
            'R': float(n.R),
            'C': float(n.C)
        }

    for s in module.al_synapses:
        g.add_edge(s.from_, s.to, type='directed',
                   attr_dict={
                       'model': 'AlphaSynapse',
                       'name': s.id,
                       'class': int(s.class_),
                       'ar': float(s.ar),
                       'ad': float(s.ad),
                       'gmax': float(s.gmax),
                       'reverse': float(s.reverse)
                   })
    for s in module.pgg_synapses:
        g.add_edge(s.from_, s.to, type='directed',
                   attr_dict={
                       'model': 'power_gpot_gpot',
                       'name': s.id,
                       'class': int(s.class_),
                       'slope': float(s.slope),
                       'threshold': float(s.threshold),
                       'power': float(s.power),
                       'saturation': float(s.saturation),
                       'delay': float(s.delay),
                       'reverse': float(s.reverse),
                       'conductance': True if s.conductance == 'true' else False
                   })

    i = nx.Graph()
    for p in module.interface.ports:
        i.add_node(p.identifier)
        i.node[p.identifier] = {
            'interface': int(p.interface),
            'io': p.io,
            'type': p.type
            }

    return g, i

def graph_to_nml_module(g, i, id):
    """
    Convert a module expressed as NetworkX graphs into Neurokernel NeuroML.

    Parameters
    ----------
    g : networkx.DiGraph
        Directed graph containing a module's neurons and synapses
    i : networkx.Graph
        Undirected graph containing a module's interface ports.
    id : str
        Module identifier.

    Returns
    -------
    module : neurokernel.neuroml.Module
        Module instance.
    """

    module = Module(id=id)

    for n in g.nodes():
        attr_dict = g.node[n]
        if attr_dict['model'] == 'MorrisLecar':
            ml = MLNeuron(id=attr_dict['name'],
                          extern=attr_dict['extern'],
                          public=attr_dict['public'],
                          spiking=attr_dict['spiking'],
                          V1=attr_dict['V1'],
                          V2=attr_dict['V2'],
                          V3=attr_dict['V3'],
                          V4=attr_dict['V4'],
                          phi=attr_dict['phi'],
                          offset=attr_dict['offset'],
                          init_v=attr_dict['initV'],
                          initn=attr_dict['initn'])
            module.ml_neurons.append(ml)
        elif attr_dict['model'] == 'LeakyIAF':
            lif = LifNeuron(id=n,
                            extern=attr_dict['extern'],
                            public=attr_dict['public'],
                            spiking=attr_dict['spiking'],
                            V=attr_dict['V'],
                            Vr=attr_dict['Vr'],
                            Vt=attr_dict['Vt'],
                            R=attr_dict['R'],
                            C=attr_dict['C'])
            module.lif_neurons.append(lif)
    for s in g.edges():
        attr_dict = g.edge[s[0]][s[1]]
        if attr_dict['model'] == 'AlphaSynapse':
            al = AlSynapse(id=attr_dict['name'],
                           class_=attr_dict['class'],
                           ar=attr_dict['ar'],
                           ad=attr_dict['ad'],
                           gmax=attr_dict['gmax'],
                           reverse=attr_dict['reverse'])
            module.al_synapses.append(al)
        elif attr_dict['model'] == 'power_gpot_gpot':
            pgg = PGGSynapse(id=attr_dict['name'],
                             class_=attr_dict['class'],
                             slope=attr_dict['slope'],
                             threshold=attr_dict['threshold'],
                             power=attr_dict['power'],
                             saturation=attr_dict['saturation'],
                             delay=attr_dict['delay'],
                             reverse=attr_dict['reverse'],
                             conductance=attr_dict['conductance'])
            module.pgg_synapses.append(pgg)

    interface = Interface()
    for p in i.nodes():
        attr_dict = i.node[p]
        port = Port(identifier=p,
                    interface=attr_dict['interface'],
                    io=attr_dict['io'],
                    type=attr_dict['type'])
        interface.ports.append(port)
    module.interface = interface

    return module
