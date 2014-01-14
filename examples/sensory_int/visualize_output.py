
import numpy as np
import neurokernel.LPU.utils.visualizer as vis
import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                        'true':True, 'True':True}

V = vis.visualizer()

conf_vis_input = {'type':'image','clim':[0, 0.032],
                  'shape':[32,24],'ids':[range(32*24)]}
conf_olf_input = {'type':'waveform','ids':[[0]]}

V.add_LPU('data/vision_input.h5', LPU='Vision')
V.add_plot(conf_vis_input, 'input_Vision')

V.add_LPU('data/olfactory_input.h5', LPU='Olfaction')
V.add_plot(conf_olf_input, 'input_Olfaction')

V.add_LPU('integrate_output_spike.h5', './data/integrate.gexf.gz',
          'Coincidence Detection')
V.add_plot({'type':'raster','ids': {0:range(8)},
            'yticks': range(1,9), 'yticklabels': range(8)},
           'Coincidence Detection', 'Output')

V._update_interval = 50
V.rows = 3
V.cols = 1
V.fontsize = 18
V.out_filename = 'sensory_int_output.avi'
V.codec = 'libtheora'
V.dt = 0.0001
V.xlim = [0,1.4]
V.run()


