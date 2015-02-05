import vision_configuration as vc
import numpy as np
np.random.seed(10000)

lamina = vc.Lamina(24, 32, 'neuron_types_lamina.csv', 'synapse_lamina.csv', None)
lamina.create_cartridges()
lamina.connect_cartridges()
lamina.create_non_columnar_neurons()
lamina.connect_composition_II()
lamina.connect_composition_I()
lamina.add_selectors()
lamina.export_to_gexf('lamina.gexf.gz')

medulla = vc.Medulla(24, 32, 'neuron_types_medulla.csv', 'synapse_medulla.csv', 'synapse_medulla_other.csv')
medulla.create_cartridges()
medulla.connect_cartridges()
medulla.create_non_columnar_neurons()
medulla.connect_composition_I()
medulla.connect_composition_II()
medulla.connect_composition_III()
medulla.add_selectors()
medulla.export_to_gexf('medulla.gexf.gz')


