import numpy as np
import neurokernel.LPU.utils.visualizer as vis

V = vis.visualizer()

V.add_LPU('medulla_output_spike.h5', './config_files/medulla.gexf.gz','lobula')
V.add_plot({'type':'raster'}, 'lobula', 'HS',9216)

config1 = {}
config1['type'] = 'image'
config1['shape'] = [32,24]
config1['clim'] = [-0.55,-0.2]
config2 = config1.copy()
config2['clim'] = [-0.52,-0.51]



V.add_LPU('lamina_output_gpot.h5', './config_files/lamina.gexf.gz','lamina')
V.add_plot(config1, 'lamina', 'R1')
V.add_plot(config2, 'lamina', 'L1')

V.add_LPU('medulla_output_gpot.h5', './config_files/medulla.gexf.gz','medulla')
V.add_plot(config2, 'medulla', 'T5a')


V._update_interval = 50
V.out_filename = 'output_int.avi'
V.codec = 'h264'

V.run()

'''
for key,configs in V._config.iteritems():
    print key
    for config in configs:
        print config['title']
        print np.min(V._data[key][config['ids'][0],:])
        print np.max(V._data[key][config['ids'][0],:])
'''
