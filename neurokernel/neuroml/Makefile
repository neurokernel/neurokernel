# To regenerate the NeuroML parser after modifying neurokernel.xsd, run make in
# this directory. The generateDS.py tool may be obtained from
# https://pypi.python.org/pypi/generateDS/

all:
	generateDS.py -o nml.py --use-getter-setter=none --silence --user-methods=helper_methods neurokernel.xsd

