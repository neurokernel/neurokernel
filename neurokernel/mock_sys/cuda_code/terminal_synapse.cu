// Used 14 registers, 64 bytes cmem[0], 4 bytes cmem[16]

#define N_synapse %(n_synapse)d

__global__ void update_terminal_synapse(double* buffer, int buffer_ld, int current, int delay_steps, int* pre_neuron, double* conductance, double* thres, double* slope, double* power, double* saturation, int* delay, double* mem_tmp)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    
    int pre;
    double mem;
    int dl;
    int col;
    
    for(int i = tid; i < N_synapse; i += total_threads)
    {
        pre = pre_neuron[i];
        dl = delay[i];
        col = current - dl;
        if(col < 0)
        {
            col = delay_steps + col;
        }
        
        mem = buffer[col * buffer_ld + pre];
        mem_tmp[i] = mem;
        //conductance[i] = fmax(0.0, fmin(saturation[i], slope[i] * pow(mem - thres[i], power[i])));
        conductance[i] = fmin(saturation[i], slope[i] * pow(fmax(0.0, mem - thres[i]), power[i]));
    }

}
