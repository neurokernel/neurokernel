from random import uniform
import numpy as np
from collections import OrderedDict
from lxml import etree
from abc import ABCMeta, abstractmethod, abstractproperty

class Synapse(object):
    __metaclass__ = ABCMeta

    _default_attr_dict = OrderedDict((
        ("model","string"),
        ("name","string"),
        ("reverse","float"),
        ("class","integer"),
        ("conductance","boolean")))

    _attr_dict = OrderedDict({})

    def __init__(self, id, pre_neu, post_neu):
        self.id = id
        self.pre_neu = pre_neu
        self.post_neu = post_neu

    @property
    def commonAttr(self):
        return super(type(self),self)._attr_dict

    @classmethod
    def getAttr(cls):
        return OrderedDict({})

    @classmethod
    def getGEXFattr(cls,etree_element):
        """
        Generate GXEF attributes

        First, add the common attribute, then add attribute from all subclasses
        """
        cls._attr_dict = OrderedDict(cls._default_attr_dict)
        for scls in cls.__subclasses__():
            for k,v in scls.getAttr().items():
                if k not in cls._attr_dict:
                    cls._attr_dict[k] = v
                else:
                    assert(cls._attr_dict[k] == v)

        for i,(k,v) in enumerate(cls._attr_dict.items()):
            etree.SubElement( etree_element, "attribute",\
                id=str(i), type=v, title=k )

    def toGEXF(self,etree_element):
        # BUG: can not get model attribute
        edge = etree.SubElement( etree_element, "edge", id=str(self.id),
            source=str(self.pre_neu.id),
            target=str(self.post_neu.id) if not isinstance(self.post_neu, Synapse) else 'synapse-'+ str(self.post_neu.id))
        return edge

class DummySynapse(Synapse):
    """
    Dummy-Synapse
    """
    def __init__(self, name=None, id=None, pre_neu=None, post_neu=None):
        super(DummySynapse, self).__init__(id, pre_neu, post_neu)
        self.name = name

    def prepare(self,dt=0.):
        pass

    def update(self,dt):
        pass

    def show(self):
        pass

    def setattr(self,**kwargs):
        """
        A wrapper of python built-in setattr(). self is returned.
        """
        for kw, val in kwargs.items():
            setattr( self, kw, val )
        return self

    def toGEXF(self,etree_element):
        comm_attr = self.commonAttr.keys()
        edge = super(DummySynapse, self).toGEXF(etree_element)
        attr = etree.SubElement( edge, "attvalues" )
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index("model")),"value":"DummySynapse"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index("name")), "value":self.name})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index("class")),"value":"2"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index("conductance")),"value":"false"})
        return edge

class AlphaSynapse(Synapse):
    """
    Alpha-Synapse
    """

    _default_attr_dict = OrderedDict((
        ("gmax","float"),
        ("ar","float"),
        ("ad","float")))

    def __init__(self, name=None, id=None, reverse=-0.06, ar=400., ad=400., gmax=1.,\
                 pre_neu=None,post_neu=None, rand=0.):
        super(AlphaSynapse, self).__init__(id, pre_neu, post_neu)
        self.name = name
        self.reverse = reverse*uniform(1.-rand,1.+rand) # the reverse potential
        self.ar = ar*uniform(1.-rand,1.+rand)           # the rise rate of the synaptic conductance
        self.ad = ad *uniform(1.-rand,1.+rand)          # the drop rate of the synaptic conductance
        self.gmax = gmax*uniform(1.-rand,1.+rand)       # maximum conductance
        self.I = 0.
        self.a = np.zeros(3,)       # a stands for alpha

    def prepare(self,dt=0.):
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
        comm_attr = self.commonAttr.keys()
        edge = super(AlphaSynapse, self).toGEXF(etree_element)
        attr = etree.SubElement( edge, "attvalues" )
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('model')), "value":"AlphaSynapse"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('name')), "value":str(self.name)})
        #if not np.isnan(self.reverse):
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('reverse')), "value":str(self.reverse)})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('class')), "value":"0"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('conductance')), "value":"true" if not np.isnan(self.reverse) else "false" })
        for att in ("gmax","ar","ad"):
            etree.SubElement( attr, "attvalue",\
                attrib={"for":str(comm_attr.index(att)), "value":str(getattr(self,att)) })

    @classmethod
    def getAttr(cls):
        return cls._default_attr_dict

class PreInhSynapse(Synapse):
    """
    Presynaptic-Inhibitory Synapse
    """
    _default_attr_dict = OrderedDict((
        ("gmax","float"),
        ("ar","float"),
        ("ad","float")))

    def __init__(self, name=None, id=None, reverse=-0.06, ar=400., ad=400., gmax=1.,\
                 pre_neu=None,post_neu=None, rand=0.):
        super(PreInhSynapse, self).__init__(id, pre_neu, post_neu)
        self.name = name
        self.reverse = reverse*uniform(1.-rand,1.+rand) # the reverse potential
        self.ar = ar*uniform(1.-rand,1.+rand)           # the rise rate of the synaptic conductance
        self.ad = ad *uniform(1.-rand,1.+rand)          # the drop rate of the synaptic conductance
        self.gmax = gmax*uniform(1.-rand,1.+rand)       # maximum conductance
        self.I = 0.
        self.a = np.zeros(3,)       # a stands for alpha

    def prepare(self,dt=0.):
        pass

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
        comm_attr = self.commonAttr.keys()
        edge = super(PreInhSynapse, self).toGEXF(etree_element)
        attr = etree.SubElement( edge, "attvalues" )
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('model')), "value":"AlphaSynapsePre"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('name')), "value":str(self.name)})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('reverse')), "value":str(self.reverse)})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('class')), "value":"0"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('conductance')), "value":"true"})
        for att in ("gmax","ar","ad"):
            etree.SubElement( attr, "attvalue",\
                attrib={"for":str(comm_attr.index(att)), "value":str(getattr(self,att)) })

    @classmethod
    def getAttr(cls):
        return cls._default_attr_dict
