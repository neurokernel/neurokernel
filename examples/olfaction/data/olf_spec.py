"""
Olfaction specification for NeuroKernel
"""
from random import uniform
import numpy as np
from collections import OrderedDict
from lxml import etree
from abc import ABCMeta, abstractmethod, abstractproperty
import gzip
from neuron import *
from synapse import *

Odor_databas       = {'Acetone':[ 1, -10,  29,   0, -25, 38,
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
    'op_syn_para':{'gmax':0.005,'reverse':0.,'ar':400.,'ad':100.},
    'type':'osn_spike_rate'
    }


class LocalNeuron:
    def __init__(self, name, pre_gl_list, post_gl_list=None, \
                para=dict(),pre_para=dict(), post_para=dict(), database = Hallem06 ):
        """
        Inter-Glomerular Local Neuron
        """
        self.neu = LeakyIAF(
            name   = name,
            V0     = database['pn_para']['V0'],
            Vr     = database['pn_para']['Vr'],
            Vt     = database['pn_para']['Vt'],
            R      = database['pn_para']['R'],
            C      = database['pn_para']['C'],
            public = False,
            extern = False)

        self.syn_list = []
        # assuming that LN-GL connectivity is reciprical
        for gl in pre_gl_list:
            # OSN->LN synapse
            for osn in gl.osn_list:
                self.syn_list.append( AlphaSynapse(
                    name     = str('%s-%s'% (osn.name,self.neu.name)),
                    gmax     = database['op_syn_para']['gmax'],
                    reverse  = database['op_syn_para']['reverse'],
                    ar       = database['op_syn_para']['ar'],
                    ad       = database['op_syn_para']['ad'],
                    pre_neu  = osn,
                    post_neu = self.neu))
        # Assuming that LN-GL connectivity is reciprical if post_gl is None
        if not post_gl_list:
            post_gl_list = pre_gl_list
        for gl in post_gl_list:
            # LN->OSN synapse
            for syn in gl.syn_list:
                if "pn" in syn.name:
                    self.syn_list.append( AlphaSynapse(
                        name     = str('%s=%s'% (self.neu.name,syn.name)),
                        gmax     = database['op_syn_para']['gmax'],
                        reverse  = database['op_syn_para']['reverse'],
                        ar       = database['op_syn_para']['ar'],
                        ad       = database['op_syn_para']['ad'],
                        pre_neu  = self.neu,
                        post_neu = syn))

class Glomerulus:
    """
    Glomerulus in the Antenna lobe of Drosophila olfactory system
    """
    def __init__(self,al,idx=None,name=None,database=None,osn_type=None,\
                 osn_num=25,pn_num=3,rand=0.,al_ref=None):
        self.idx = idx
        self.name = name
        self.osn_type = osn_type
        self.osn_num = osn_num
        self.pn_num = pn_num
        self.rand = rand
        self.al_ref = al
        if database:
            self.setNeuron(database)

    def setNeuron(self, database, osn_type=None):
        assert( database is not None )

        al_name = 'al' if self.al_ref is None else self.al_ref.name
        # overide osn_type if a new one is given
        if osn_type:
            self.osn_type = osn_type
        # check osn_type is supported by the odor database; If not,
        osn = [o for o in self.osn_type if o in database['osn']]
        if len(osn) > 0:
            rate = database['osn_rate'][osn[0]]
            self.osn_type = osn[0]
            C = 1./rate/database['osn_para']['C']/np.log(database['osn_para']['Vr']/database['osn_para']['Vt'])
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
                extern=False,
                rand=self.rand,
                selector=str('/%s/%d/pn/%d' % (al_name, self.idx, i))))

        self.osn_list = [] # initialize the osn list
        self.syn_list = []
        self.rece_list = []
        for i in xrange(self.osn_num):
            self.osn_list.append(LeakyIAF(
                name=str('osn_%s_%d' % (self.osn_type,i)),
                V0=database['osn_para']['V0'],
                Vr=database['osn_para']['Vr'],
                Vt=database['osn_para']['Vt'],
                R=database['osn_para']['R'],
                C=C,
                public=False,
                extern=True,
                rand=self.rand))
            self.rece_list.append(Receptor(
                name=str('rece_%s_%d' % (self.osn_type,i)),
                selector=str('/%s/%d/rece/%d' % (al_name, self.idx, i))))
            self.syn_list.append(DummySynapse(
                name=str('rece-%s' % (self.osn_list[i].name,)),
                pre_neu=self.rece_list[i],
                post_neu=self.osn_list[i]))
            # setup synpases from the current OSN to each of PNs
            for j in xrange(self.pn_num):
                self.syn_list.append(PreInhSynapse(
                    name=str('%s-%s'% (self.osn_list[i].name,\
                                         self.pn_list[j].name)),
                    gmax=database['op_syn_para']['gmax'],
                    reverse=database['op_syn_para']['reverse'],
                    ar=database['op_syn_para']['ar'],
                    ad=database['op_syn_para']['ad'],
                    pre_neu=self.osn_list[i],
                    post_neu=self.pn_list[j],
                    rand=self.rand))
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
    def __init__(self, name=None, anatomy_db=None, odor_db=None, gl_name=None):
        self.name = name or 'al'
        self.anatomy_db = anatomy_db
        self.odor_db = odor_db
        self.gl_name = gl_name
        self.neu_list = None
        self.syn_list = None
        self.ln_list = []
        self.gl_list = []

    def setGlomeruli(self, anatomy_db=None, odor_db=None, gl_name=None, rand=0.):
        if anatomy_db is not None: self.anatomy_db = anatomy_db
        if odor_db is not None: self.odor_db = odor_db
        if gl_name is not None: self.gl_name = gl_name
        for i,gl in enumerate(self.gl_name):
            #assert( gl is in self.database['gl'] )
            self.gl_list.append( Glomerulus(
                al=self,
                name=gl,
                idx=i,
                database=self.odor_db,
                osn_type=self.anatomy_db['gl'][gl],
                rand=rand))
        self.gl_dict = {gl.name:gl for gl in self.gl_list}

    def addLN(self, ln_name, pre_gl_list, post_gl_list = [], pre_para = {}, post_para = {}):
            assert( set(pre_gl_list) <= set(self.gl_name.keys()))
            assert( set(post_gl_list) <= set(self.gl_name.keys()))
            pre_gl_list = {self.gl_dict[x] for x in pre_gl_list}
            post_gl_list = {self.gl_dict[x] for x in post_gl_list}
            self.ln_list.append(LocalNeuron(
                ln_name,
                pre_gl_list,
                post_gl_list))

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
        Neuron.getGEXFattr( node )
        # add edge(synapse) attributes
        edge = etree.SubElement( graph, "attributes", attrib={"class":"edge"} )
        Synapse.getGEXFattr( edge )
        # convert all neurons into GEXF
        nodes = etree.SubElement( graph, "nodes" )
        for neu in self.neu_list: neu.toGEXF( nodes )
        # convert all synapse into GEXF
        if not no_synapse:
            edges = etree.SubElement( graph, "edges" )
            for syn in self.syn_list: syn.toGEXF( edges )
        # write to file if filename is given
        if filename is not None:
            with gzip.open(filename, 'w') as f:
                etree.ElementTree(root).write(f, pretty_print = True,
                                              xml_declaration = True, encoding="utf-8")
        return root

    def _getAllNeuList(self):
        self.neu_list = []
        # stack OSN onto the neuron list
        for gl in self.gl_list:
            for neu in gl.osn_list:
                self.neu_list.append( neu.setattr( id=len(self.neu_list)))
        # stack PN onto the neuron list
        for gl in self.gl_list:
            for neu in gl.pn_list:
                self.neu_list.append( neu.setattr( id=len(self.neu_list)))
        # stack receptors of each glomeruli onto the receptor list
        for gl in self.gl_list:
            for rece in gl.rece_list:
                self.neu_list.append( rece.setattr(id=len(self.neu_list)))
        # stack OSN onto the neuron list
        for ln in self.ln_list:
            self.neu_list.append( ln.neu.setattr( id=len(self.neu_list)))

    def _getAllSynList(self):
        self.syn_list = []
        for gl in self.gl_list:
            for syn in gl.syn_list:
                self.syn_list.append( syn.setattr( id=len(self.syn_list)))
        for neu in self.ln_list:
            for syn in neu.syn_list:
                self.syn_list.append( syn.setattr( id=len(self.syn_list)))
