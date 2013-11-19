#!/usr/bin/env python

"""
Create GEXF file detailing antennal lobe neurons and synapses.
"""

from olf_spec import *

al = AntennalLobe(anatomy_db=Luo10, odor_db=Hallem06, gl_name=Luo10['gl'])
al.setGlomeruli()
al.toGEXF('antennallobe.gexf')
