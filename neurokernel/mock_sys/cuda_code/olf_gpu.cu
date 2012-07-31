#define MAX_THREAD 1024
struct LeakyIAF
{
    double V;
    double Vr;
    double Vt;
    double tau;
    double R;
    int    num;
    int    offset;
};

struct AlphaSynNL
{
    long   neu_idx;
    double neu_coe;
};

struct AlphaSyn
{
    double g;
    double alpha[3];
    double gmax;
    double taur;
    double sign;
    int    num;    // number of innerved neuron 
    int    offset; // Offset in the array
};

#define update_neu_V( neuron, I, bh, ic, spk )  \
{                                               \
    neuron.V = neuron.V*bh + I*ic;              \
    spk = 0;                                    \
    if( neuron.V >= neuron.Vt )                 \
    {                                           \
        neuron.V = neuron.Vr;                   \
        spk = 1;                                \
    }                                           \
}

#define neu_thread_copy( neuron, bh, ic, n_s_list,               \
                         all_neu_syn_list, cid, I_map, eid )     \
{                                                                \
    bh = exp( -dt/neuron.tau );                                  \
    ic = neuron.R*( 1.0-bh );                                    \
    n_s_list = all_neu_syn_list + neuron.offset;                 \
    cid = I_map;                                                 \
    eid = cid;                                                   \
}

#define syn_thread_copy( synapse, num, tr, gmax, sign,           \
                         s_n_list, all_syn_neu_list )            \
{                                                                \
    num  = synapse.num;                                          \
    gmax = synapse.gmax;                                         \
    tr   = synapse.taur;                                         \
    sign = synapse.sign;                                         \
    s_n_list = all_syn_neu_list + synapse.offset;                \
}

#define update_syn_G( g, num, g_old, g_new, gmax, tr,            \
                      sign, s_n_list,  spk_list  )               \
{                                                                \
    /* Update g(t) */                                            \
    g_new[0] = g_old[0] + dt*g_old[1];                           \
    if( g_new[0] < 0.0 ) g_new[0] = 0.0;                         \
                                                                 \
    /* Update g'(t) */                                           \
    g_new[1] = g_old[1] + dt*g_old[2];                           \
    for( int j=0; j<num; ++j)                                    \
        if( spk_list[ s_n_list[j].neu_idx ] )                    \
            g_new[1] += (s_n_list[j].neu_coe);                   \
                                                                 \
    /* Update g"(t) */                                           \
    g_new[2] = (-2.0*g_old[1] - tr*g_old[0])*tr;                 \
    /* Copy g_old to g_new */                                    \
    for( int j=0; j<3; ++j ) g_old[j] = g_new[j];                \
    g = sign*gmax*g_new[0];                                      \
    __syncthreads();                                             \
}

#define update_neu_I( neuron, I, synapse, n_s_list, post_g )     \
{                                                                \
    post_g = 0.0;                                                \
                                                                 \
    for( int j=0; j<neuron.num; ++j )                            \
        post_g += (synapse + n_s_list[j])->g;                    \
    I = I + post_g*( neuron.V-neuron.Vr );                       \
}

__global__ void gpu_run_dt( 
                    double dt, 
                    int neu_num, int syn_num,
                    LeakyIAF *neuron, int *neu_syn_list,
                    AlphaSyn *synapse, AlphaSynNL *syn_neu_list,
                    int *I_ext_map, 
                    double *I_ex,
                    int *spike_list
                )
{
    // In this per-dt-run global function, status of one neuron and one 
    // synapse are computed.

    // thread id
    const int tid = threadIdx.x + threadIdx.y*blockDim.x;
    // Unit id, an unit is a neuron or a synapse.
    const int uid = tid + blockIdx.x*gridDim.x; 

    // Compute variables for further usage
    

    double I=0.0, ic, bh;  // For updating  
    int* n_s_list;         // Synapses list of a neuron
    double alpha_new[3];   //
    double post_g;         //
    AlphaSynNL *s_n_list;  // Pre-synaptic inhibitory Neuron list of a synapse

    // Update Synapse Status
    if( uid < syn_num ){
        s_n_list = syn_neu_list + synapse[uid].offset;   
        update_syn_G( synapse[uid].g, synapse[uid].num, synapse[uid].alpha, 
                      alpha_new, synapse[uid].gmax, synapse[uid].taur,
                      synapse[uid].sign, s_n_list, spike_list );
     } // if
    
    // Update Neuron External Current ( only for sensory neurons )
    //if()

    // Update Neuron Post-synaptic Current ( summed with external current )
    if( uid < neu_num )
        update_neu_I( neuron[uid], I, synapse, n_s_list, post_g);

    // Update Neuron Membrane Voltage, spikes are recorded if neuron fires
    if( uid < neu_num ){
        bh = exp( -dt/neuron[uid].tau );
        ic = neuron[uid].R;
        update_neu_V( neuron[uid], I, bh, ic, spike_list[uid] );
    } // if
} 

__global__ void gpu_run( int N, double dt, 
                         int neu_num, LeakyIAF *neuron, 
                         int *neu_syn_list,
                         int syn_num, AlphaSyn *synapse, 
                         AlphaSynNL *syn_neu_list,
                         int *spike_list,
                         int *I_ext_map, 
                         int I_ext_num, int I_ext_len,
                         double *I_ext
                       )
{
    // Constant for neuron update
    //__shared__ double BH[MAX_THREAD];
    //__shared__ double IC[MAX_THREAD]; 

    const int tid = threadIdx.x+threadIdx.y*blockDim.x;
    // unit idx; unit is either neuron or synapse
    const int uid = tid + blockIdx.x * gridDim.x; 
    int sid = uid;                // spike idx, updated per dt
    int eid = 0;                  // external current idx, updated per dt
    int cid = 0;

    // local copy of neuron parameters
    int* n_s_list;
    double bh, ic, post_g, I=0.0;
    int *dt_spk_list = spike_list;
    if( uid < neu_num )
        neu_thread_copy( neuron[uid], bh, ic, n_s_list, neu_syn_list,
                         cid, I_ext_map[uid], eid );

    // local copy of synapse parameters
    int num;
    double g_new[3],tau_r, gmax, sign;
    double g_old[3] = {0, 0, 0};
    AlphaSynNL *s_n_list;
    
    if( uid < syn_num )
        syn_thread_copy( synapse[uid], num, tau_r, gmax, sign, 
                         s_n_list, syn_neu_list );

    
    // Simulation Loop
    for( int i = 0; i<N; ++i )
    {
        // Update Neuron Membrane Voltage
        if( uid < neu_num )
            update_neu_V( neuron[uid], I, bh, ic, spike_list[sid] );

        // Update Synapse Status
        if( uid < syn_num )
            update_syn_G( synapse[uid].g, num, g_old, g_new, gmax, tau_r,
                          sign, s_n_list, dt_spk_list );
        
        // Update External Current( only for sensory neuron )
        I = 0.0;
        if( cid!=-1 && i < I_ext_len ) I = I_ext[eid];

        // Update Synaptic Current 
        if( uid < neu_num )
            update_neu_I( neuron[uid], I, synapse, n_s_list, post_g);

        // Update Spike idx, external current idx, and dt-spike array address
        sid += neu_num;
        eid += I_ext_num;
        dt_spk_list += neu_num;
    }
}

__global__ void spk_rate( int N, float dt, float overlap,
                         int neu_num, int *spk_list, double *rate
                       )
{
    const int tid = threadIdx.x+threadIdx.y*blockDim.x;
    const int num = int(overlap/dt);
    
    if(tid>=neu_num) return;
    int pre_half, post_half;
    int spk_idx = tid;
    int idx = tid;
    pre_half = 0;

    for(int i = 0; i<num; ++i)
    {
        pre_half += spk_list[spk_idx];
        spk_idx += neu_num;
    }
    for(int i = num; i<N-num; i+=num, idx+=neu_num )
    {
        post_half = 0;
        for(int j = i; j<i+num; ++j)
        {
            post_half += spk_list[spk_idx];
            spk_idx += neu_num;
        }
        rate[idx] = float( post_half  )/overlap/2.0;
        idx += neu_num;
        pre_half = post_half;
    }
}
