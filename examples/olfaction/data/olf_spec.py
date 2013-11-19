#!/usr/bin/env python

"""
Olfaction model specification.
"""

import numpy as np
from random import random
from lxml import etree
import pdb

Odor_database       = {'Acetone':[ 1, -10,  29,   0, -25, 38,
                                 -7, -10,   8, -16,  11,  3,
                                 -6,  -3, 130,  -7,  27, -5,
                                 -4,  -5,  28,  18,  -9, -1],
                      'Methanol':[10, 22, 25,  5, -16, 55,
                                  -3, 35, 38, -2,  14, 13,
                                 -21,  9, 18,  3,   3, 20,
                                   3, 14, 17,  6,  -8,  5],
                      'Ecstasy':[100000,10000,0,0,0,0,
                                   0,0,0,0,0,0,
                                   0,0,0,0,0,0,
                                   0,0,0,0,0,0],
                      'None':[]}
Luo10 = {
    'gl':{'VM7':('Or42a',),'VM2':('Or43b',),'VM3':('Or9a',),'VA1lm':('Or47b',),
          'VM5d':('Or98b', 'Or85b'),'VC2':('Or71a',),'DL1':('Or10a',),
          'VC3':('Or35a',),'DL3':('Or65a', 'Or65c', 'Or65b'),
          'VC1':('Or33c', 'Or85e'),'VC4':('Or67c',),'DL5':('Or7a',),
          'DL4':('Or49a', 'Or85f'),'VA2':('Or92a',),'VA3':('Or67b',),
          'VM5v':('Or98a',),'VA6':('Or82a',),'VA4':('Or85d',),'VA5':('Or49b',),
          'D':('Or69aA', 'Or69aB'),'VA1d':('Or88a',),'DA4m':('Or2a',),
          'DA4l':('Or43a',),'VA7l':('Or46aA',),'DA1':('Or67d',),
          'DA2':('Or56a', 'Or33a'),'DA3':('Or23a',),'DC2':('Or13a',),
          'DC3':('Or83c',),'DC1':('Or19b', 'Or19a'),'l':('Or59c',),
          'DM1':('Or42b',),'DM2':('Or22a', 'Or22b'),'DM3':('Or47a', 'Or33b'),
          'DM4':('Or59b',),'DM5':('Or85a',),'DM6':('Or67a',),
          'DC4':(),'DL2d':(),'DL2v':(),'DL6':(),'DP1l':(),'VM4':(),
          'DP1m':(),'VA7m':(),'VL1':(),'VL2a':(),'VL2p':(),'V':(),
          'VM1':(),'VM6':(),'VP1':(),'VP2':(),'VP3':(),'VC3M':()},
    'osn':{'Or1a':None,'Or2a':'DA4m','Or7a':'DL5','Or9a':'VM3','Or10a':'DL1',
           'Or13a':'DC2','Or19a':'DC1','Or19b':'DC1','Or22a':'DM2',
           'Or22b':'DM2','Or22c':None,'Or23a':'DA3','Or24a':None,'Or30a':None,
           'Or33a':'DA2','Or33b':'DM3','Or33c':'VC1','Or35a':'VC3',
           'Or42a':'VM7','Or42b':'DM1','Or43a':'DA4l','Or43b':'VM2',
           'Or45a':None,'Or45b':None,'Or46aA':'VA7l','Or47a':'DM3',
           'Or47b':'VA1lm','Or49a':'DL4','Or49b':'VA5','Or56a':'DA2',
           'Or59a':None,'Or59b':'DM4','Or59c':'l','Or63a':None,'Or65a':'DL3',
           'Or65b':'DL3','Or65c':'DL3','Or67a':'DM6','Or67b':'VA3',
           'Or67c':'VC4','Or67d':'DA1','Or69aA':'D','Or69aB':'D','Or71a':'VC2',
           'Or74a':None,'Or82a':'VA6','Or83a':None,'Or83b':None,'Or83c':'DC3',
           'Or85a':'DM5','Or85b':'VM5d','Or85c':None,'Or85d':'VA4',
           'Or85e':'VC1','Or85f':'DL4','Or88a':'VA1d','Or92a':'VA2',
           'Or94a':None,'Or94b':None,'Or98a':'VM5v','Or98b':'VM5d'},
    'type':'anatomy'
    }

Hallem06 = {
    'osn':('Or2a','Or7a','Or9a','Or10a','Or19a','Or22a','Or23a','Or33b',
           'Or35a','Or43a','Or43b','Or47a','Or47b','Or49b','Or59b','Or65a',
           'Or67a','Or67c','Or82a','Or85a','Or85b','Or85f','Or88a','Or98a'),
    'osn_rate':{ 'Or2a': 8,'Or7a':17,'Or9a': 3,'Or10a':14,'Or19a':29,
                 'Or22a': 4,'Or23a': 9,'Or33b':25,'Or35a':17,'Or43a':21,
                 'Or43b': 2,'Or47a': 1,'Or47b':47,'Or49b': 8,'Or59b': 2,
                 'Or65a':18,'Or67a':11,'Or67c': 6,'Or82a':16,'Or85a':14,
                 'Or85b':13,'Or85f': 7,'Or88a':26,'Or98a':12},
    'osn_para':{'Vr':-0.07,'Vt':-0.025,'R':1,'C':0.07,'V0':-0.05},
    'pn_para':{'Vr':-0.07,'Vt':-0.025,'R':1,'C':0.07,'V0':-0.05},
    'op_syn_para':{'gmax':1,'reverse':-80,'ar':400,'ad':400},
    'type':'osn_spike_rate'
    }

class AlphaSynapse:
    """
    Alpha-Synapse
    """
    def __init__(self, name=None, id=None, reverse=-60, ar=400, ad=400, gmax=1,\
                 pre_neu=None,post_neu=None):
        self.id = id
        self.name = name
        self.reverse = reverse # the reverse potential
        self.ar = ar           # the rise rate of the synaptic conductance
        self.ad = ad           # the drop rate of the synaptic conductance
        self.gmax = gmax
        self.I = 0
        self.a = np.zeros(3,)  # a stands for alpha
        self.pre_neu = pre_neu
        self.post_neu = post_neu

    def prepare(self,dt=0):
        pass

    def update(self,dt):
        """
        Update the synapse state variables
        """
        new_a = np.zeros(3)
        # update a
        new_a[0] = max([0., self.a[0]+dt*self.a[1]])
        # update a'
        new_a[1] = self.a[1]+dt*self.a[2]
        if self.pre_neu.isSpiking:
            new_a[1] += self.ar*self.ad
        # update a"
        new_a[2] = -(self.ar+self.ad)*new_a[1] - self.ar*self.ad*new_a[0]
        # copy new_a to a
        self.a[:] = new_a
        # update synaptic current
        self.I = self.gmax*self.a[0]*(self.post_neu.V-self.reverse)

    def show(self):
        print "%s gmax:%.2f reverse:%.2f ar:%.2f ad:%.2f" % \
              (self.name,self.gmax,self.Vr,self.ar,self.ad)

    def setattr(self,**kwargs):
        """
        A wrapper of python built-in setattr(). self is returned.
        """
        for kw, val in kwargs.items():
            setattr( self, kw, val )
        return self

    def toGEXF(self,etree_element):
        edge = etree.SubElement( etree_element, "edge", id=str(self.id),
            source=str(self.pre_neu.id), target=str(self.post_neu.id))
        attr = etree.SubElement( edge, "attvalues" )
        etree.SubElement(attr,"attvalue",attrib={"for":"0","value":"AlphaSynapse"})
        for i,att in enumerate( ("name","reverse","ar","ad","gmax") ):
            etree.SubElement( attr, "attvalue",\
                attrib={"for":str(i+1), "value":str(getattr(self,att)) })
        etree.SubElement(attr,"attvalue",attrib={"for":"6","value":"0"})
        etree.SubElement(attr,"attvalue",attrib={"for":"7","value":"true"})

    @staticmethod
    def getGEXFattr(etree_element):
        """
        generate GXEF attributes
        """
        def_type = etree.SubElement( etree_element, "attribute",\
                       id="0", type="string", title="type" )
        etree.SubElement( def_type, "default" ).text = "AlphaSynapse"
        etree.SubElement( etree_element, "attribute",\
            id="1", type="string", title="name" )
        for (i,attr) in enumerate( ("reverse","ar","ad","gmax") ):
            etree.SubElement( etree_element, "attribute",\
                id=str(i+2), type="float", title=attr )
        etree.SubElement( etree_element, "attribute", id="6",\
            type="integer", title="class" )
        etree.SubElement( etree_element, "attribute", id="7",
            type="boolean", title="conductance")

class LeakyIAF:
    """
    Leaky Integrated-and-Fire Neuron
    """

    def __init__(self,id=None,name=None,V0=0,Vr=-50,Vt=-20,R=1,C=1,\
                 syn_list=None,public=True,input=True):
        self.id = id
        self.name = name
        self.V = V0
        self.Vr = Vr
        self.Vt = Vt
        self.R = R
        self.C = C

        # For GEXF
        self.public = public
        self.input = input

        self.isSpiking = False
        if syn_list is not None:
            self.syn_list = syn_list
        else:
            self.syn_list = []

    def prepare(self,dt=0):
        """
        prepare intermediate variables for simulation

        - self.bh : a variable used by Exponential Euler method
        """
        self.bh = np.exp( -dt/self.R/self.C )

    def update(self, I_ext=0):
        """
        Update the neuron state

        This function is split into three steps: 1) compute external current 2)
        update membrane voltage 3) spike detection. The reverse potential of
        every synapse is assumed to be the same as the resting potential of the
        neuron. The exponential Euler method is applied to perform the
        numerical integration.

        input:

        """

        # Compute the total external current
        I_syn = 0
        for syn in self.syn_list:
            I_syn += syn.I
        self.I = I_ext + I_syn

        # Update the membrane voltage
        self.V = self.V*self.bh + self.R*self.I*(1-self.bh)
        # Spike detection
        if self.V >= self.Vt:
            self.V = self.Vr
            self.isSpiking = True
        else:
            self.isSpiking = False

    def show(self):
        """
        print the neuron parameters
        """
        print "%s->%s %f %f %f %f %f" % \
              (self.pre_neu.name,self.post_neu.name,\
              self.Vr,self.Vt,self.R,self.C)

    def setattr(self,**kwargs):
        """
        A wrapper of python built-in setattr(). self is returned.
        """
        for kw, val in kwargs.items():
            setattr( self, kw, val )
        return self

    def toGEXF(self,etree_element):
        node = etree.SubElement( etree_element, "node", id=str(self.id) )
        attr = etree.SubElement( node, "attvalues" )
        etree.SubElement(attr,"attvalue",attrib={"for":"0","value":"LeakyIAF"})
        for i,att in enumerate( ("name","Vr","Vt","R","C") ):
            etree.SubElement( attr, "attvalue",\
                attrib={"for":str(i+1), "value":str(getattr(self,att)) })
        etree.SubElement( attr, "attvalue", attrib={"for":"6", "value":"true" })
        etree.SubElement( attr, "attvalue", attrib={"for":"7", "value":"true" if self.public else "false" })
        etree.SubElement( attr, "attvalue", attrib={"for":"8", "value":"true" if self.input else "false" })

    @staticmethod
    def getGEXFattr(etree_element):
        """
        generate GXEF attributes

        """
        def_type = etree.SubElement( etree_element, "attribute",\
                       id="0", type="string", title="type" )
        etree.SubElement( def_type, "default" ).text = "LeakyIAF"
        etree.SubElement( etree_element, "attribute",\
            id="1", type="string", title="name" )
        for (i,attr) in enumerate( ("Vr","Vt","R","C") ):
            etree.SubElement( etree_element, "attribute",\
                id=str(i+2), type="float", title=attr )
        for (i,attr) in enumerate( ("spiking","public","input") ):
            etree.SubElement( etree_element, "attribute",\
                id=str(i+6), type="boolean", title=attr )

class Glomerulus:
    """
    Glomerulus in the Antenna lobe of Drosophila olfactory system
    """
    def __init__(self,idx=None,name=None,database=None,osn_type=None,\
                 osn_num=25,pn_num=3):
        self.idx = idx
        self.name = name
        self.osn_type = osn_type
        self.osn_num = osn_num
        self.pn_num = pn_num
        if database:
            self.setNeuron(database)

    def setNeuron(self, database, osn_type=None):
        assert( database is not None )
        # overide osn_type if a new one is given
        if osn_type: self.osn_type = osn_type
        # check osn_type is supported by the odor database; If not,
        osn = [o for o in self.osn_type if o in database['osn']]
        if len(osn) > 0:
            rate = database['osn_rate'][osn[0]]
            self.osn_type = osn[0]
            C = 1/rate   #TODO
        else:
            self.osn_type = 'default'
            C = database['osn_para']['C']
        # setup PNs
        self.pn_list = []
        for i in xrange(self.pn_num):
            self.pn_list.append(LeakyIAF(
                name=str('%s_pn_%d'% (self.name,i)),
                V0=database['pn_para']['V0'],
                Vr=database['pn_para']['Vr'],
                Vt=database['pn_para']['Vt'],
                R=database['pn_para']['R'],
                C=database['pn_para']['C'],
                public=True,
                input=False))
        self.osn_list = [] # initialize the osn list
        self.syn_list = []
        for i in xrange(self.osn_num):
            self.osn_list.append(LeakyIAF(
                name=str('osn_%s_%d' % (self.osn_type,i)),
                V0=database['osn_para']['V0'],
                Vr=database['osn_para']['Vr'],
                Vt=database['osn_para']['Vt'],
                R=database['osn_para']['R'],
                C=C,
                public=True,
                input=True))
            # setup synpases from the current OSN to each of PNs
            for j in xrange(self.pn_num):
                self.syn_list.append(AlphaSynapse(
                    name=str('%s-%s'% (self.osn_list[i].name,\
                                         self.pn_list[j].name)),
                    gmax=database['op_syn_para']['gmax'],
                    reverse=database['op_syn_para']['reverse'],
                    ar=database['op_syn_para']['ar'],
                    ad=database['op_syn_para']['ad'],
                    pre_neu=self.osn_list[i],
                    post_neu=self.pn_list[j]))
                self.pn_list[j].syn_list.append(self.syn_list[-1])
        return self

    def prepare(self,dt):
        for osn in self.osn_list:
            osn.prepare(dt)
        for pn in self.pn_list:
            pn.prepare(dt)
        for syn in self.syn_list:
            syn.prepare(dt)

    def update(self,dt):
        for osn in self.osn_list:
            osn.update(dt)
        for pn in self.pn_list:
            pn.update(dt)
        for syn in self.syn_list:
            syn.update(dt)

class AntennalLobe():
    def __init__(self, anatomy_db=None, odor_db=None, gl_name=None):
        self.anatomy_db = anatomy_db
        self.odor_db = odor_db
        self.gl_name = gl_name
        self.neu_list = None
        self.syn_list = None

    def setGlomeruli(self, anatomy_db=None, odor_db=None, gl_name=None):
        if anatomy_db is not None: self.anatomy_db = anatomy_db
        if odor_db is not None: self.odor_db = odor_db
        if gl_name is not None: self.gl_name = gl_name
        self.gl_list = []
        for gl in self.gl_name:
            #assert( gl is in self.database['gl'] )
            self.gl_list.append( Glomerulus(
                name=gl,
                database=self.odor_db,
                osn_type=self.anatomy_db['gl'][gl]))


    def setLN():
        pass

    def toGEXF(self, filename=None, no_synapse=False):
        if self.neu_list is None: self._getAllNeuList()
        if self.syn_list is None: self._getAllSynList()

        root = etree.Element("gexf",xmlns="http://www.gexf.net/1.2draft", version="1.2")
        # add graph element
        graph = etree.SubElement( root, "graph", defaultedgetype="directed" )
        # add node(neuron) attributes
        node = etree.SubElement( graph, "attributes", attrib={"class":"node"} )
        LeakyIAF.getGEXFattr( node )
        # add edge(synapse) attributes
        edge = etree.SubElement( graph, "attributes", attrib={"class":"edge"} )
        AlphaSynapse.getGEXFattr( edge )
        # convert all neurons into GEXF
        nodes = etree.SubElement( graph, "nodes" )
        for neu in self.neu_list: neu.toGEXF( nodes )
        # convert all synapse into GEXF
        if not no_synapse:
            edges = etree.SubElement( graph, "edges" )
            for syn in self.syn_list: syn.toGEXF( edges )
        # write to file if filename is given
        if filename is not None:
            etree.ElementTree(root).write( filename, pretty_print = True,
                xml_declaration = True, encoding="utf-8" )
        return root

    def _getAllNeuList(self):
        self.neu_list = []
        # stack OSN into the neuron list
        for gl in self.gl_list:
            for neu in gl.osn_list:
                self.neu_list.append( neu.setattr( id=len(self.neu_list)))
        # stack PN into the neuron list
        for gl in self.gl_list:
            for neu in gl.pn_list:
                self.neu_list.append( neu.setattr( id=len(self.neu_list)))

    def _getAllSynList(self):
        self.syn_list = []
        for gl in self.gl_list:
            for syn in gl.syn_list:
                self.syn_list.append( syn.setattr( id=len(self.syn_list)))
