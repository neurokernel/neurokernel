__global__ void alpha_synapse(
    int num,
    %(type)s dt,
    int *spike,
    int *Pre,
    %(type)s *Ar,
    %(type)s *Ad,
    %(type)s *Gmax,
    %(type)s *a0,
    %(type)s *a1,
    %(type)s *a2,
    %(type)s *cond )
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    int pre;
    %(type)s ar,ad,gmax;
    %(type)s old_a[3];
    %(type)s new_a[3];

    for( int i=tid; i<num; i+=tot_threads ){
        // copy data from global memory to register
        ar = Ar[i];
        ad = Ad[i];
        pre = Pre[i];
        gmax = Gmax[i];
        old_a[0] = a0[i];
        old_a[1] = a1[i];
        old_a[2] = a2[i];

        // update the alpha function
        new_a[0] = fmax( 0., old_a[0] + dt*old_a[1] );
        new_a[1] = old_a[1] + dt*old_a[2];
        if( spike[pre] )
            new_a[1] += ar*ad;
        new_a[2] = -( ar+ad )*old_a[1] - ar*ad*old_a[0];

        // copy data from register to the global memory
        a0[i] = new_a[0];
        a1[i] = new_a[1];
        a2[i] = new_a[2];
        cond[i] = new_a[0]*gmax;
    }
    return;
}
