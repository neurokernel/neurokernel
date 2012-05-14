// Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0],
// 308 bytes cmem[2], 28 bytes cmem[16]

#define NVAR 2
#define NNEU 64 //NROW * NCOL
#define NTYPE %(ntype)d

#define V_L (-0.5)
#define V_Ca 1
#define V_K (-0.7)
#define g_Ca 1.1
#define g_K 2
#define g_L 0.5


__device__ __constant__ %(type)s V_1[NTYPE];
__device__ __constant__ %(type)s V_2[NTYPE];
__device__ __constant__ %(type)s V_3[NTYPE];
__device__ __constant__ %(type)s V_4[NTYPE];
__device__ __constant__ %(type)s Tphi[NTYPE];
__device__ __constant__ int neuron_start[NTYPE];
__device__ __constant__ %(type)s offset[NTYPE];


__device__ int search_for_type(int cart_id)
{
    int low = 0, high = NTYPE, mid;
    while(low < high)
    {
        mid = ((low+high)>>1);
        if(cart_id >= neuron_start[mid])
        {
            low = mid + 1;
        }else
        {
            high = mid;
        }
    }
    return low-1;
}

__device__ %(type)s compute_n(%(type)s V, %(type)s n, int type)
{
    %(type)s dn = Tphi[type] * cosh(( V - V_3[type]) / (V_4[type]*2)) * 0.5 * ((1 + tanh((V - V_3[type]) / V_4[type])) - n);
    return dn;
}

__device__ %(type)s compute_V(%(type)s V, %(type)s n, %(type)s I, int type)
{
    %(type)s m_inf = 0.5 * (1+tanh((V - V_1[type])/V_2[type]));
    %(type)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca * m_inf * (V - V_Ca) + offset[type]);
    return dV;
}


__global__ void
hhn_euler_multiple(%(type)s* V, %(type)s* n, %(type)s* V_buffer, int num_neurons, %(type)s* I_pre, %(type)s dt, int nsteps)
{
    int neuron_id = threadIdx.x;
    int vid = threadIdx.y;
    int bid = blockIdx.x;
    int cart_id = bid * NNEU + neuron_id;
    
    int type = search_for_type(cart_id);
    
    __shared__ %(type)s tmpstates[NVAR][NNEU];

    double I;

    if(cart_id < num_neurons)
    {
        if(vid == 0)
        {
            tmpstates[0][neuron_id] = V[cart_id];
            I = I_pre[min(cart_id, num_neurons)];
        }else if (vid == 1)
        {
            tmpstates[1][neuron_id] = n[cart_id];
        }
    }
    
    __syncthreads();
    
    %(type)s d, tmp;
    tmp = tmpstates[vid][neuron_id];
    
    for(int i = 0; i < nsteps; ++i)
    {
        if(vid == 1)
        {
            d = compute_n(tmpstates[0][neuron_id], tmp, type);
        }
        
        if(vid == 0)
        {
            d = compute_V(tmpstates[0][neuron_id], tmp, I, type);
        }
        
        tmp += d * dt;
        
        __syncthreads();
        
        tmpstates[vid][neuron_id] = tmp; //would have to change it to 1 neuron per thread model
        
        __syncthreads();
    }
    
    if(vid == 0)
    {
        if(cart_id < num_neurons)
        {
            V[cart_id] = tmpstates[0][neuron_id];
            V_buffer[cart_id] = tmpstates[0][neuron_id];
        }
    }else if(vid == 1)
    {
        if(cart_id < num_neurons)
        {
            n[cart_id] = tmpstates[1][neuron_id];
        }
    }
    
}