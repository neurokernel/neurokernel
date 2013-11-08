// %(type)s and %(nneu)d must be replaced using Python string foramtting
#define NNEU %(nneu)d

__global__ void leaky_iaf(
    int neu_num,
    %(type)s dt,
    int      *spk,
    %(type)s *V,
    %(type)s *I,
    %(type)s *Vt,
    %(type)s *Vr,
    %(type)s *R,
    %(type)s *C)
{
    int bid = blockIdx.x;
    int nid = bid * NNEU + threadIdx.x;

    %(type)s v,i,r,c;

    if( nid < neu_num ){
        v = V[nid];
        i = I[nid];
        r = R[nid];
        c = C[nid];

        // update v
        %(type)s bh = exp( -dt/r/c );
        v = v*bh + r*i*(1.0-bh);

        // spike detection
        spk[nid] = 0;
        if( v >= Vt[nid] ){
            v = Vr[nid];
            spk[nid] = 1;
        }

        V[nid] = v;
    }
    return;
}
