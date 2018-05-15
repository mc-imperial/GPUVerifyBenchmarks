//pass
//--num_groups=128 --local_size=64

#include "../common.h"

__kernel void
chk1D_512(__global T2* work, int half_n_cmplx, __global int* fail)
{
    int i, tid = get_local_id(0); 
    int blockIdx = get_group_id(0) * 512 + tid; 
    T2 a[8], b[8];
    
    work = work + blockIdx; 

    for (i = 0; i < 8; i++) {
        a[i] = work[i*64];
    }
    
    for (i = 0; i < 8; i++) {
        b[i] = work[half_n_cmplx+i*64];
    }

    for (i = 0; i < 8; i++) {
        if (a[i].x != b[i].x || a[i].y != b[i].y) {
            *fail = 1;
        }
    }
}
