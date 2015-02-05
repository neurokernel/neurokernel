#Maps from NeuroML element space (Key) to python object space (value)
#The class names are essentially correct, the instance names need converting
#also attributes need fixing

import lxml
from lxml import objectify
import re
from config import variables
import csv
import sys

sys.setrecursionlimit(1500)

def remove_curlies(string):
    return re.sub("{.*}","",string)

def to_lowercase_with_underscores(string):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def to_camelback(string):
    string_list = list(string)
    i = 0
    for m in re.finditer('_', string):
        underscore_index = m.end()
        string_list[underscore_index - i] = string_list[underscore_index - i].capitalize()
        string_list.pop(underscore_index - (1 + i))
        i += 1
    string = ''.join(string_list)

    return str(string)

def traverse_doc(queue,rename):
    """Recursive function to traverse the nodes of a tree in
    breadth-first order.

    The first argument should be the tree root; children
    should be a function taking as argument a tree node and
    returning an iterator of the node's children.
    """
    if len(queue) > 0:
        node = queue.pop()
        children = node.getchildren()
        rename(node)
        queue = queue + children
        traverse_doc(queue,rename)
    else:
        return None

def pluralize(noun):                            
    if re.search('[sxz]$', noun):             
        return re.sub('$', 'es', noun)        
    elif re.search('[^aeioudgkprt]h$', noun):
        return re.sub('$', 'es', noun)       
    elif re.search('[^aeiou]y$', noun):      
        return re.sub('y$', 'ies', noun)     
    else:                                    
        return noun + 's'

def _node_to_python(node):

    pluralize_flag = 'maxOccurs' in node.attrib

    for attribute in node.attrib:
        nml_attribute = node.attrib.pop(attribute)
        if nml_attribute[0].islower():
            renamed_attribute = to_lowercase_with_underscores(nml_attribute)
            if pluralize_flag:
                renamed_attribute = pluralize(renamed_attribute)
            NameTable[nml_attribute] = renamed_attribute

filename = variables['schema_name']

import StringIO
import process_includes
outfile = StringIO.StringIO()

infile = open(filename, 'r')
process_includes.process_include_files(infile, outfile,
        inpath=filename)
infile.close()
outfile.seek(0)
doc = objectify.parse(outfile)
root = doc.getroot()
queue = [root]

NameTable = {}

traverse_doc(queue,_node_to_python)

#filtering routine, need to get a better way to extract these, asked on Stack Overflow
import keyword
disallowed_keywords = keyword.kwlist
for keyword in disallowed_keywords:
    try:
        NameTable.pop(keyword)
    except:
        pass

NameTable['morphology'] = 'morphology' #overriding change to
#"morphologies" because it only applies outside of a cell - not a very
#elegant solution
NameTable['gateHHtauInf'] = 'gate_hh_tau_infs'
NameTable['ionChannelHH'] = 'ion_channel_hh'
NameTable['gateHHrates']  = 'gate_hh_rates'
NameTable['gateHHtauInf'] = 'gate_hh_tau_infs'

NameTable['ionChannel'] = 'ion_channel'
NameTable['ionChannelHH'] = 'ion_channel_hhs'



NameTable['basePyNNCell'] = 'basePyNNCell'
NameTable['basePyNNIaFCell'] = 'basePyNNIaFCell'
NameTable['basePyNNIaFCondCell'] = 'basePyNNIaFCondCell'
NameTable['tau_syn_E'] = 'tau_syn_E'
NameTable['tau_syn_I'] = 'tau_syn_I'

NameTable['e_rev_E'] = 'e_rev_E'
NameTable['e_rev_I'] = 'e_rev_I'
NameTable['e_rev_Na'] = 'e_rev_Na'
NameTable['e_rev_K'] = 'e_rev_K'

NameTable['gbar_K'] = 'gbar_K'
NameTable['gbar_Na'] = 'gbar_Na'

NameTable['IF_curr_alpha'] = 'IF_curr_alpha'
NameTable['IF_curr_exp'] = 'IF_curr_exp'
NameTable['IF_cond_alpha'] = 'IF_cond_alpha'
NameTable['IF_cond_exp'] = 'IF_cond_exp'

NameTable['extracellularProperties'] = 'extracellular_properties'
NameTable['intracellularProperties'] = 'intracellular_properties'
NameTable['biophysicalProperties'] = 'biophysical_properties'

print("NameTable is as follows:")
print(NameTable)

# print("Saving NameTable to csv file")
# writer = csv.writer(open('name_table.csv', 'wb'))
# for key, value in NameTable.items():
#   writer.writerow([key, value])

# print ("Saving name changes table to csv file")
# changes_writer = csv.writer(open('changed_names.csv','wb'))
# for key in NameTable:
#     value = NameTable[key]
#     if key != value:
#         changes_writer.writerow([key,value])
