import csv
import numpy as np

synapse_list = []
dtypes = [np.dtype('S10'),np.dtype('S10'),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.int32), np.dtype(np.double),
                  np.dtype(np.int32), np.dtype(np.int32),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.int32)]
with open('synapse_types.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile)
    synapse_field_name = reader.next()
    n_entry = len(synapse_field_name)
    for row in reader:
        tmp = [dtypes[i].type(row[i]) for i in range(n_entry)]
        synapse_list.append(tuple(tmp))

num_synapse_types = len(synapse_list)

for i in range(num_synapse_types):
    print "    {'prename':'"+synapse_list[i][0]+"', 'postname':'"+synapse_list[i][1]+"', 'model':'power_gpot_gpot_sig',"
    print "    'cart':"+str(None if synapse_list[i][10] == 0 else synapse_list[i][10]) +", 'reverse':"+str(synapse_list[i][5])+", 'delay':"+str(synapse_list[i][4])+","
    print "    'threshold':"+str(synapse_list[i][8])+", 'slope':"+str(synapse_list[i][3])+"', 'power':"+str(synapse_list[i][2])+", 'saturation':"+str(synapse_list[i][9])+","
    print "    'scale':"+str(synapse_list[i][6])+", 'mode':0},"

