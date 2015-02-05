#!/usr/bin/env python

"""
Demo of how to generate and save a Neurokernel emulation in NeuroML.
"""

from neurokernel.neuroml import NeuroMLDocument
from neurokernel.neuroml import NeurokernelDoc
from neurokernel.neuroml import MLNeuron
from neurokernel.neuroml import AlSynapse
from neurokernel.neuroml import Interface
from neurokernel.neuroml import Module
from neurokernel.neuroml import Pattern
from neurokernel.neuroml import PatternConnection
from neurokernel.neuroml import Port
from neurokernel.neuroml import ModuleConnection

from neurokernel.neuroml import write

def create_test_nml():
    doc = NeurokernelDoc(id='TestDoc')

    # First module:
    interface = Interface()
    interface.ports.append(Port(interface=0, 
                                io='out', type='gpot',
                                identifier='/lam[0]'))
    interface.ports.append(Port(interface=0,
                                io='out', type='gpot',
                                identifier='/lam[1]'))  
    interface.ports.append(Port(interface=0,
                                io='out', type='gpot',
                                identifier='/lam[2]'))

    module = Module(id='lam')
    module.interface = interface
    module.ml_neurons.append(MLNeuron(id='ML0',
                                      extern=True, public=True, spiking=False,
                                      V1=1.0, V2=2.0, V3=1.5, V4=2.3,
                                      phi=1.2, offset=-2.6,
                                      init_v=2.0, initn=1.0))
    module.ml_neurons.append(MLNeuron(id='ML1',
                                      extern=True, public=True, spiking=False,
                                      V1=4.2, V2=2.2, V3=0.1, V4=0.5,
                                      init_v=3.0, initn=2.0,
                                      phi=2.0, offset=-1.5))
    module.ml_neurons.append(MLNeuron(id='ML2',
                                      extern=True, public=True, spiking=False,
                                      V1=0.3, V2=0.5, V3=1.2, V4=5.1,
                                      init_v=1.1, initn=0.0,
                                      phi=3.0, offset=-2.25))
    module.al_synapses.append(AlSynapse(id='AL0', 
                                        class_= 0,
                                        ar=1.5, ad=3.0,
                                        gmax=1, reverse=-7.0,
                                        from_='ML0', to='ML1'))
    module.al_synapses.append(AlSynapse(id='AL1', 
                                        class_= 0,
                                        ar=1.0, ad=2.0,
                                        gmax=2.3, reverse=-6.0,
                                        from_='ML1', to='ML2'))
    doc.modules.append(module)

    # Second module:
    interface = Interface()
    interface.ports.append(Port(interface=0, 
                                io='in', type='gpot',
                                identifier='/med[0]'))
    interface.ports.append(Port(interface=0,
                                io='in', type='gpot',
                                identifier='/med[1]'))  
    interface.ports.append(Port(interface=0,
                                io='in', type='gpot',
                                identifier='/med[2]'))

    module = Module(id='med')
    module.interface = interface
    module.ml_neurons.append(MLNeuron(id='ML0',
                                      extern=True, public=True, spiking=False,
                                      V1=1.3, V2=0.5, V3=2.5, V4=1.5,
                                      phi=4.0, offset=-3.0,
                                      init_v=1.0, initn=2.0))
    module.ml_neurons.append(MLNeuron(id='ML1',
                                      extern=True, public=True, spiking=False,
                                      V1=0.2, V2=3.2, V3=2.0, V4=0.5,
                                      phi=4.0, offset=-3.5,
                                      init_v=3.0, initn=2.0,))
    module.ml_neurons.append(MLNeuron(id='ML2',
                                      extern=True, public=True, spiking=False,
                                      V1=0.3, V2=0.5, V3=1.0, V4=6.0,
                                      init_v=1.1, initn=0.3,
                                      phi=2.3, offset=-2.4))
    module.al_synapses.append(AlSynapse(id='AL0', 
                                     class_= 0,
                                     ar=1.2, ad=4.0,
                                     gmax=2, reverse=-2.0,
                                     from_='ML0', to='ML1'))
    module.al_synapses.append(AlSynapse(id='AL1', 
                                     class_= 0,
                                     ar=2.0, ad=1.0,
                                     gmax=3.3, reverse=-3.0,
                                     from_='ML1', to='ML2'))
    module.al_synapses.append(AlSynapse(id='AL2', 
                                     class_= 0,
                                     ar=1.0, ad=3.0,
                                     gmax=1.0, reverse=-2.0,
                                     from_='ML2', to='ML0'))
    doc.modules.append(module)

    # Create pattern:
    interface = Interface()
    interface.ports.append(Port(interface=0, 
                                io='in', type='gpot',
                                identifier='/lam[0]'))
    interface.ports.append(Port(interface=0,
                                io='in', type='gpot',
                                identifier='/lam[1]'))  
    interface.ports.append(Port(interface=0,
                                io='in', type='gpot',
                                identifier='/lam[2]'))
    interface.ports.append(Port(interface=1, 
                                io='out', type='gpot',
                                identifier='/med[0]'))
    interface.ports.append(Port(interface=1,
                                io='out', type='gpot',
                                identifier='/med[1]'))  
    interface.ports.append(Port(interface=1,
                                io='out', type='gpot',
                                identifier='/med[2]'))

    pattern = Pattern(id='pat')
    pattern.interface = interface
    pattern.connections.append(PatternConnection(from_='/lam[0]',
                                                 to='/med[1]'))
    pattern.connections.append(PatternConnection(from_='/lam[0]',
                                                 to='/med[2]'))
    pattern.connections.append(PatternConnection(from_='/lam[1]',
                                                 to='/med[0]'))
    doc.patterns.append(pattern)

    # Connect modules with pattern:
    mod_connection = ModuleConnection(m0='lam', m1='med',
                                      pat='pat', int0=0, int1=1)
    doc.connections.append(mod_connection)

    return doc

doc = create_test_nml()
write(doc, 'emulation.nml')
