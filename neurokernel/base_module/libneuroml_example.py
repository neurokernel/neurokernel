"""

Example to build a full spiking IaF network throught libNeuroML & save it as
XML & validate it.

"""

from neuroml.v2.NeuroMLDocument import NeuroMLDocument
from neuroml.v2 import IaFCell
from neuroml.v2 import Network
from neuroml.v2 import ExpOneSynapse
from neuroml.v2 import Population
from neuroml.v2 import PulseGenerator
from neuroml.v2 import ExplicitInput
from neuroml.v2 import SynapticConnection
from random import random
from neuroml.loaders import NeuroMLLoader

###########################   Build the network   #############################
nmlDoc = NeuroMLDocument(id = "Olfactory_System")
# Specifying all cells
IaF_osn = IaFCell(id = "osn", C = "0.03 nF", thresh = "-25mV", reset = "-70mV", leakConductance = "1 uS", leakReversal = "-50mV")

IaF_pn = IaFCell(id = "osn", C = "0.03 nF", thresh = "-25mV", reset = "-70mV",
                   leakConductance = "1 uS", leakReversal = "-50mV")

# Inserting neurons into NML specification
nmlDoc.add_iafCell(IaF_osn)
nmlDoc.add_iafCell(IaF_pn)


# Specifying synapses
syn_osn = ExpOneSynapse(id = "syn_osn", gbase = "65nS", erev = "0mV",
                     tauDecay = "3ms")

syn_pn = ExpOneSynapse(id = "syn_pn", gbase = "65nS", erev = "0mV",
                     tauDecay = "3ms")

# Inserting synapses into NML specification
nmlDoc.add_expOneSynapse(syn_osn)
nmlDoc.add_expOneSynapse(syn_pn)


# Specifying the network
olfactory_sys = Network(id = "Olfactory_System")
# Inserting the network into NML specification
nmlDoc.add_network(olfactory_sys)


# Specifying all populations of neurons for antenna and antenna lobe
osn_pop_list = {'2a':8, '7a':17, '9a':3, '10a':14, '19a':29, '22a':4, '23a':9,
                '33b':25, '35a':17, '43a':21, '43b':2, '47a':1, '47b':47,
                '49b':8, '59b':2, '65a':18, '67a':11, '67c':6, '82a':16,
                '85b':13, '85f':7, '88a':26}
gl_pop_list = {'DA4m':10, 'DL5':10, 'VM3':10, 'DL1':10, 'DC1':10, 'DM2':10,
               'DA3':10, 'DM5':10, 'VC3':10, 'DA41':10, 'VM2':10, 'DM3':10,
               'VA1lm':10, 'VA5':10, 'DM4':10, 'DL3':10, 'DM6':10, 'VC4':10,
               'VA6':10, 'VM5d':10, 'VC1':10, 'VA1d':10}
an_left = range(len(osn_pop_list))
al_left = range(len(gl_pop_list))
an_right = range(len(osn_pop_list))
al_right = range(len(gl_pop_list))

if (len(osn_pop_list) != len(gl_pop_list)):
    raise Exception("The number of populations in AN and AL must be the same")

lengh = len(osn_pop_list)
for pop in range(0, lengh):
    an_left[pop] = Population(id = osn_pop_list.keys()[pop] + "_l",
                              component = IaF_osn.id,
                              size = osn_pop_list.values()[pop])
    olfactory_sys.add_population(an_left[pop])
    al_left[pop] = Population(id = gl_pop_list.keys()[pop] + "_l",
                              component = IaF_pn.id,
                              size = gl_pop_list.values()[pop])
    olfactory_sys.add_population(al_left[pop])
    an_right[pop] = Population(id = osn_pop_list.keys()[pop] + "_r",
                              component = IaF_osn.id,
                              size = osn_pop_list.values()[pop])
    olfactory_sys.add_population(an_right[pop])
    al_right[pop] = Population(id = gl_pop_list.keys()[pop] + "_r",
                              component = IaF_pn.id,
                              size = gl_pop_list.values()[pop])
    olfactory_sys.add_population(al_right[pop])


# Connections from each OSN to each Glomerulus, in this case we are just
# connecting Antenna's OSN populations to AL's GL population
lengh = len(an_left)
for pop in range(0, lengh):
    num_osn = osn_pop_list.values()[pop]
    for pre in range(0, num_osn):
        num_pn = gl_pop_list.values()[pop]
        for post in range(0, num_pn):
            olfactory_sys.add_synapticConnection(
                 SynapticConnection(fromxx = "%s[%i]" % (an_left[pop].id, pre),
                                    synapse = syn_osn.id,
                                    to = "%s[%i]" % (al_left[pop].id, post)))
            olfactory_sys.add_synapticConnection(
                 SynapticConnection(fromxx = "%s[%i]" % (an_left[pop].id, pre),
                                    synapse = syn_osn.id,
                                    to = "%s[%i]" % (al_right[pop].id, post)))
            olfactory_sys.add_synapticConnection(
                 SynapticConnection(fromxx = "%s[%i]" % (an_right[pop].id, pre),
                                    synapse = syn_osn.id,
                                    to = "%s[%i]" % (al_right[pop].id, post)))
            olfactory_sys.add_synapticConnection(
                 SynapticConnection(fromxx = "%s[%i]" % (an_right[pop].id, pre),
                                    synapse = syn_osn.id,
                                    to = "%s[%i]" % (al_left[pop].id, post)))


# Saving into XML file
newnmlfile = "olfactory_system.xml"
nmlDoc.write_neuroml(newnmlfile)


###########################  Save to file & validate  #########################
from lxml import etree
from urllib import urlopen

schema_file = urlopen("http://neuroml.svn.sourceforge.net/viewvc/" + \
                      "neuroml/NeuroML2/Schemas/NeuroML2/NeuroML_v2alpha.xsd")
xmlschema_doc = etree.parse(schema_file)
xmlschema = etree.XMLSchema(xmlschema_doc)

print "Validating %s against %s" % (newnmlfile, schema_file.geturl())

doc = etree.parse(newnmlfile)
xmlschema.assertValid(doc)
print "It's valid!"


###########################  Loading file  ####################################
loaded_doc = NeuroMLLoader.load_neuroml(newnmlfile)

