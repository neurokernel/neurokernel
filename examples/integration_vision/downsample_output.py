import neurokernel.LPU.utils.simpleio as si

a=si.read_array('lamina_output_gpot.h5')
# downsample 100 times and store as loutput.h5
b=a[0::100,:]
si.write_memory_to_file(b,'loutput_nk.h5')

# The internal data storage file for medulla is medulla_output.h5
# read out full history of medulla neuron output
a=si.read_array('medulla_output_gpot.h5')
# downsample 100 times and store as moutput.h5
b=a[0::100,:]
si.write_memory_to_file(b,'moutput_nk.h5')
