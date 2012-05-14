import pycuda.driver as cuda
import neuralnet as nn
import atexit
import numpy as np
import pycuda.gpuarray as garray
from simpleio import *
import parray
import time
import drawim as dr
import getpass

dev = 1

cuda.init()
ctx = cuda.Device(dev).make_context()
atexit.register(ctx.pop)
    
np.random.seed(0)

dt = 1e-4
dur = 1.0
Nt = 10000#int(dur/dt)
t = np.arange(0,1,dt)

num_dendrites = read_file('example/n_dendrites.h5').astype(np.int32)
num_neurons = num_dendrites.size
pre_neuron = read_file('example/pre.h5').astype(np.int32)
num_synapse = pre_neuron.size
post_neuron = read_file('example/post.h5').astype(np.int32)
thres = read_file('example/threshold.h5')
slope = read_file('example/slope.h5')
power = read_file('example/power.h5')
saturation = read_file('example/saturation.h5')
delay = read_file('example/delay.h5')
reverse = read_file('example/reverse.h5')

start_idx = read_file('example/start_idx.h5').astype(np.int32)
num_types = start_idx.size
V_1 = read_file('example/V1.h5')
V_2 = read_file('example/V2.h5')
V_3 = read_file('example/V3.h5')
V_4 = read_file('example/V4.h5')
Tphi = read_file('example/T.h5')
offset = read_file('example/offset.h5')

#intial condition at resting potential
V = read_file('example/initV.h5')
n = read_file('example/initn.h5')

network = nn.network(num_types, num_neurons, 24*32, start_idx, num_dendrites, num_synapse, pre_neuron, post_neuron, thres, slope, power, saturation, delay, reverse, dt, V, n, V_1, V_2, V_3, V_4, Tphi, offset, 6)

#input video
h_I_ext = np.zeros(10000, 4608)

I_ext = parray.to_gpu(h_I_ext)

out = np.empty((Nt,num_neurons), np.double)

#turn this on to allow visual output
prin = False

if prin:
    from matplotlib import pylab as p
    import matplotlib
    matplotlib.interactive(True)
    
    fig = p.figure(figsize=(20,8))
    ax1 = p.subplot(291)
    ax2 = p.subplot(292)
    ax3 = p.subplot(293)
    ax4 = p.subplot(294)
    ax5 = p.subplot(295)
    ax6 = p.subplot(296)
    ax7 = p.subplot(297)
    ax8 = p.subplot(298)
    ax9 = p.subplot(299)
    ax10 = p.subplot(2,9,10)
    ax11 = p.subplot(2,9,11)
    ax12 = p.subplot(2,9,12)
    ax13 = p.subplot(2,9,13)
    ax14 = p.subplot(2,9,14)
    ax15 = p.subplot(2,9,15)
    ax16 = p.subplot(2,9,16)
    ax17 = p.subplot(2,9,17)
    ax18 = p.subplot(2,9,18)

    tit = fig.suptitle('0')

playstep = 100
for i in range(Nt):
    network.run_step(int(I_ext.gpudata) + I_ext.dtype.itemsize * I_ext.ld * i, out[i,:])
    if prin:
        if i % playstep == 1:
            input = h_I_ext[i,:768].reshape(32,24)
            R1 = out[i,start_idx[0]:start_idx[1]].reshape(32,24)
            L1= out[i,start_idx[6]:start_idx[7]].reshape(32,24)
                        
            L2 = out[i,start_idx[7]:start_idx[8]].reshape(32,24)
                        
            L3 = out[i,start_idx[8]:start_idx[9]].reshape(32,24)
            
            L4 = out[i,start_idx[9]:start_idx[10]].reshape(32,24)
            
            L5 = out[i,start_idx[10]:start_idx[11]].reshape(32,24)
            T1 = out[i,start_idx[11]:start_idx[12]].reshape(32,24)
            
            
            C2 = out[i,start_idx[12]:start_idx[13]].reshape(32,24)
                        
            ax1.cla()
            ax1.imshow(dr.image_trans(input), cmap = p.cm.gray, vmin=0,vmax=0.3)
            ax1.set_title('input')
            
            ax2.cla()
            ax2.imshow(dr.image_trans(R1), cmap = p.cm.gray, vmin=-0.6,vmax=-0.1)
            ax2.set_title('R1')
            ax3.cla()
            ax3.imshow(dr.image_trans(L1), cmap = p.cm.gray, vmin=-0.5-0.02,vmax=-0.5+0.02)
            ax3.set_title('L1')
            
            ax4.cla()
            ax4.imshow(dr.image_trans(L2), cmap = p.cm.gray, vmin=-0.5-0.02,vmax=-0.5+0.02)
            ax4.set_title('L2')
            
            ax5.cla()
            ax5.imshow(dr.image_trans(L3), cmap = p.cm.gray, vmin=-0.5-0.02,vmax=-0.5+0.02)
            ax5.set_title('L3')
            
            ax6.cla()
            ax6.imshow(dr.image_trans(L4), cmap = p.cm.gray, vmin=-0.5-0.02,vmax=-0.5+0.02)
            ax6.set_title('L4')
            
            ax7.cla()
            ax7.imshow(dr.image_trans(L5), cmap = p.cm.gray, vmin=-0.5-0.02,vmax=-0.5+0.02)
            ax7.set_title('L5')
            ax8.cla()
            ax8.imshow(dr.image_trans(T1), cmap = p.cm.gray, vmin=-0.5-0.02,vmax=-0.5+0.02)
            ax8.set_title('T1')
            
            ax9.cla()
            ax9.imshow(dr.image_trans(C2), cmap = p.cm.gray, vmin=-0.5-0.02,vmax=-0.5+0.02)
            ax9.set_title('C2')
            
            pix = 228
            
            ax10.cla()
            ax10.plot(t[:i],h_I_ext[:i,pix])
            ax10.set_xlim(0,1)
            ax10.set_ylim(-0.01,0.3)
            ax10.set_title('input')
            ax11.cla()
            ax11.plot(t[:i],out[:i,pix])
            ax11.set_xlim(0,1)
            ax11.set_ylim(-0.65,-0.1)
            ax11.set_title('R1')
            ax12.cla()
            ax12.plot(t[:i],out[:i,pix+start_idx[6]])
            ax12.set_xlim(0,1)
            ax12.set_ylim(-0.5-0.02,-0.5+0.02)
            ax12.set_title('L1')
            
            ax13.cla()
            ax13.plot(t[:i],out[:i,pix+start_idx[7]])
            ax13.set_xlim(0,1)
            ax13.set_ylim(-0.5-0.02,-0.5+0.02)
            ax13.set_title('L2')
            
            ax14.cla()
            ax14.plot(t[:i],out[:i,pix+start_idx[8]])
            ax14.set_xlim(0,1)
            ax14.set_ylim(-0.5-0.02,-0.5+0.02)
            ax14.set_title('L3')
            
            ax15.cla()
            ax15.plot(t[:i],out[:i,pix+start_idx[9]])
            ax15.set_xlim(0,1)
            ax15.set_ylim(-0.5-0.02,-0.5+0.02)
            ax15.set_title('L4')
            
            ax16.cla()
            ax16.plot(t[:i],out[:i,pix+start_idx[10]])
            ax16.set_xlim(0,1)
            ax16.set_ylim(-0.5-0.02,-0.5+0.02)
            ax16.set_title('L5')
            
            ax17.cla()
            ax17.plot(t[:i],out[:i,pix+start_idx[11]])
            ax17.set_xlim(0,1)
            ax17.set_ylim(-0.5-0.02,-0.5+0.02)
            ax17.set_title('T1')
            
            ax18.cla()
            ax18.plot(t[:i],out[:i,pix+start_idx[12]])
            ax18.set_xlim(0,1)
            ax18.set_ylim(-0.5-0.02,-0.5+0.02)
            ax18.set_title('C2')
            
            tit.set_text('%d' % i)
            p.draw()
            time.sleep(0.1)
