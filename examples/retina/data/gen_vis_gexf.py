import h5py
import networkx as nx

def generate_gexf(output_file, photoreceptor_num):
    G = nx.DiGraph()

# alternative graph generation
    G.add_nodes_from(range(photoreceptor_num))
    
    for i in range(photoreceptor_num):
        name = 'photor' + str(i)
        G.node[i] = {
            'model': 'Photoreceptor',
            'name': name,
            'extern': True,  # gets input from file
            'public': True,  # it's an output neuron
            'spiking': False,
            'selector': '/ret/' + str(i),
            'num_microvilli': 30000
        }

    nx.write_gexf(G, output_file)


if __name__ == '__main__':
    # use to generate custom gexf
    # currently generate_gexf is used externally
    pass