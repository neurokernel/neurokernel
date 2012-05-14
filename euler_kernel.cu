// Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0],
// 308 bytes cmem[2], 28 bytes cmem[16]

#define NVAR 2
#define NNEU %(nneu)d //NROW * NCOL
#define NTYPE %(ntype)d

#define V_L (-0.5)
#define V_Ca 1.0
#define V_K (-0.7)
#define g_Ca 1.1
#define g_K 2.0
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
        }
        else
        {
            high = mid;
        }
    }
    return low-1;
}

__device__ %(type)s compute_n(%(type)s V, %(type)s n, int type)
{
    %(type)s n_inf = 0.5 * (1 + tanh((V - V_3[type]) / V_4[type]));
    %(type)s dn = Tphi[type] * cosh(( V - V_3[type]) / (V_4[type]*2)) *
        (n_inf - n);
    return dn;
}

__device__ %(type)s compute_V(%(type)s V, %(type)s n, %(type)s I, int type)
{
    %(type)s m_inf = 0.5 * (1+tanh((V - V_1[type])/V_2[type]));
    %(type)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca *
        m_inf * (V - V_Ca) + offset[type]);
    return dV;
}


__global__ void hhn_euler_multiple(%(type)s* g_V, %(type)s* g_n, %(type)s* V_buffer, int num_neurons, %(type)s* I_pre, %(type)s dt, int nsteps)
{
    int bid = blockIdx.x;
    int cart_id = bid * NNEU + threadIdx.x;
    
    int type = search_for_type(cart_id);

    %(type)s I, V, n;

    if(cart_id < num_neurons)
    {
        V = g_V[cart_id];
        I = I_pre[cart_id];
        n = g_n[cart_id];
    }
    
    
    
    %(type)s dV, dn;
    
    for(int i = 0; i < nsteps; ++i)
    {
        dn = compute_n(V, n, type);
        
        dV = compute_V(V, n, I, type);
        
        V += dV * dt;
        n += dn * dt;
    }
    
    
    if(cart_id < num_neurons)
    {
        g_V[cart_id] = V;
        V_buffer[cart_id] = V;
        g_n[cart_id] = n;
    }
    
}