// #Used 15 registers, 8704+0 bytes smem, 64 bytes cmem[0]

#define N 32
#define NUM_NEURONS %(num_neurons)d

__global__ void get_input(double* synapse, int* cum_num_dendrite, int* num_dendrite, double* I_pre, double* V, double* V_rev)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bid = blockIdx.x;
    
    int neuron;
    
    __shared__ int num_den[32];
    __shared__ int den_start[32];
    __shared__ double V_in[32];
    __shared__ double input[32][33];
    
    if(tidy == 0)
    {
        neuron = bid * N + tidx;
        if(neuron < NUM_NEURONS)
        {
            num_den[tidx] = num_dendrite[neuron];
            V_in[tidx] = V[neuron];
        }
    }else if(tidy == 1)
    {
        neuron = bid * N + tidx;
        if(neuron < NUM_NEURONS)
        {
            den_start[tidx] = cum_num_dendrite[neuron];
        }
    }
    
    input[tidy][tidx] = 0.0;
    
    __syncthreads();
    
    int n_den = num_den[tidy];
    int start = den_start[tidy];
    double VV = V_in[tidy];
    
    for(int i = tidx; i < n_den; i += N)
    {
        input[tidy][tidx] += synapse[start + i] * (VV - V_rev[start + i]);
    }
    
    __syncthreads();
    
    
    
    if(tidy < 8)
    {
        input[tidx][tidy] += input[tidx][tidy + 8];
        input[tidx][tidy] += input[tidx][tidy + 16];
        input[tidx][tidy] += input[tidx][tidy + 24];
    }
    
    __syncthreads();
    
    if(tidy < 4)
    {
        input[tidx][tidy] += input[tidx][tidy + 4];
    }
    
    __syncthreads();
    
    if(tidy < 2)
    {
        input[tidx][tidy] += input[tidx][tidy + 2];
    }
    
    __syncthreads();
    
    if(tidy == 0)
    {
        input[tidx][0] += input[tidx][1];
        
        if(neuron < NUM_NEURONS)
        {
            I_pre[neuron] -= input[tidx][0];
        }
    }
    
    
}
//can be improved