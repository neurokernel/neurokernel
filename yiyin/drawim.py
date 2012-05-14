import numpy as np

def image_trans(A):
    im = A.T.copy()
    I = np.zeros((343, 352),np.double)-10
    b = np.tile(np.tile(im.reshape(-1,1),(1,11)).reshape(-1,1,32*11),(1,14,1)).reshape(336,352)
    for i in range(32):
        if i % 2 == 0:
            I[:336,i*11:(i+1)*11] = b[:,i*11:(i+1)*11]
        else:
            I[7:,i*11:(i+1)*11] = b[:,i*11:(i+1)*11]
        
    return I
    
    
    
def draw_map(output, start_idx, type, frame):
    I = output[frame, start_idx[type]:start_idx[type+1]].reshape(24,32)
    return image_trans(I)