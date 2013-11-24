#!/usr/bin/env python

"""
Create GEXF file detailing antennal lobe neurons and synapses.
"""

from olf_spec import *
from random import seed

seed(200601050) 
al = AntennalLobe(anatomy_db=Luo10, odor_db=Hallem06, gl_name=Luo10['gl'])
al.setGlomeruli(rand=0.05)
al.toGEXF('antennallobe.gexf.gz')
