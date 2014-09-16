import vision_configuration as vc
import numpy as np
np.random.seed(10000)

lamina = vc.Lamina(24, 32, 'neuron_types_lamina.csv', 'synapse_lamina.csv', None)
lamina.create_cartridges()
lamina.connect_cartridges()
lamina.create_non_columnar_neurons()
lamina.connect_composition_II()
lamina.connect_composition_I()
lamina.export_to_gexf('lamina.gexf.gz')

