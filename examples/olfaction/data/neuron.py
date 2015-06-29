from random import uniform
import numpy as np
from collections import OrderedDict
from lxml import etree
from abc import ABCMeta, abstractmethod, abstractproperty

class Neuron(object):
    __metaclass__ = ABCMeta

    _default_attr_dict = OrderedDict((
                 ('model','string'),
                 ('name','string'),
                 ('spiking','float'),
                 ('public','float'),
                 ('extern','float'),
                 ('selector','string')))

    _attr_dict = OrderedDict({})

    def __init__(self):
        pass

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
        comm_attr = self.commonAttr.keys()
        node = etree.SubElement( etree_element, "node", id=str(self.id) )
        attr = etree.SubElement( node, "attvalues" )
        for i,k in enumerate(comm_attr):
            v = getattr(self, k, None)
            if v is not None and v != "":
                etree.SubElement(attr, "attvalue",
                                 attrib={"for":str(i), "value":str(v)})

class Receptor(Neuron):
    """
    Dummy receptor for setting up ports; no computation invloved in this class
    """
    def __init__(self, id=None, name=None, selector=None):
        self.id = id
        self.name = name
        self.selector = selector or ''

    def setattr(self,**kwargs):
        """
        A wrapper of python built-in setattr(). self is returned.
        """
        for kw, val in kwargs.items():
            setattr( self, kw, val )
        return self

    def toGEXF(self,etree_element):
        comm_attr = self.commonAttr.keys()
        node = etree.SubElement(etree_element, "node", id=str(self.id))
        attr = etree.SubElement(node, "attvalues")
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('model')), "value":"port_in_gpot"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('name')), "value":self.name})
        for i, x in enumerate(('spiking','public','extern')):
            etree.SubElement( attr, "attvalue", attrib={"for":str(comm_attr.index(x)), "value":"false" })
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('selector')), "value":self.selector})

class LeakyIAF(Neuron):
    """
    Leaky Integrated-and-Fire Neuron
    """

    _default_attr_dict = OrderedDict((
        ("V","float"),
        ("Vr","float"),
        ("Vt","float"),
        ("R","float"),
        ("C","float")))

    def __init__(self,id=None,name=None,V0=0.,Vr=-0.05,Vt=-0.02,R=1.,C=1.,\
                 syn_list=None,public=True,extern=True,rand=0.,\
                 selector=None,model=None):
        self.id = id
        self.name = name
        self.Vr = Vr*uniform(1.-rand,1.+rand)
        self.Vt = Vt*uniform(1.-rand,1.+rand)
        self.R = R*uniform(1.-rand,1.+rand)
        self.C = C*uniform(1.-rand,1.+rand)
        self.V = uniform(self.Vr,self.Vt)

        # For GEXF
        self.public = public
        self.extern = extern

        # For port API
        self.selector = selector or ''
        self.model = model

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

    def update(self, I_ext=0.):
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
        I_syn = 0.
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
        comm_attr = self.commonAttr.keys()
        node = etree.SubElement( etree_element, "node", id=str(self.id) )
        attr = etree.SubElement( node, "attvalues" )
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('model')), "value":"LeakyIAF"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('name')), "value":self.name})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('spiking')), "value":"true"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('public')), "value":"true" if self.public else "false"})
        etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('extern')), "value":"true" if self.extern else "false"})
        if self.selector != "":
            etree.SubElement(attr, "attvalue", attrib={"for":str(comm_attr.index('selector')), "value":self.selector})
        for i,att in enumerate( self._default_attr_dict.keys() ):
            etree.SubElement( attr, "attvalue",\
                attrib={"for":str(comm_attr.index(att)), "value":str(getattr(self,att)) })

    @classmethod
    def getAttr(cls):
        return cls._default_attr_dict
