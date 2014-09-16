import csv
import numpy as np

neuron_list = []
dtypes = [np.dtype('S10'),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double), np.dtype(np.double),
                  np.dtype(np.double)]
with open('neuron_types.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile)
    neuron_field_name = reader.next()
    n_entry = len(neuron_field_name)
    for row in reader:
        tmp = [dtypes[i].type(row[i]) for i in range(n_entry)]
        neuron_list.append(tuple(tmp))

num_neuron_types = len(neuron_list)

for i in range(num_neuron_types):
    print "    {'name':"+"'"+neuron_list[i][0]+"', 'model':'MorrisLecar_a', 'columnar':True, 'output':False,"
    print "    'extern':False, 'public':True, 'spiking':False,"
    print "    'V1':"+str(neuron_list[i][1])+", 'V2':"+str(neuron_list[i][2])+", 'V3':"+str(neuron_list[i][3])+", 'V4':"+str(neuron_list[i][4])+","
    print "    'V_l':"+str(neuron_list[i][5])+", 'V_ca':"+str(neuron_list[i][6])+", 'V_k':"+str(neuron_list[i][7])+","
    print "    'G_l':"+str(neuron_list[i][8])+", 'G_ca':"+str(neuron_list[i][9])+", 'G_k':"+str(neuron_list[i][10])+","
    print "    'phi':"+str(neuron_list[i][11])+", 'initV':"+str(neuron_list[i][12])+", 'initn':"+str(neuron_list[i][13])+", 'offset':0.0},"      
