//pass
//--gridDim=16 --blockDim=64

#include "common.h"
  
__global__ void
d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
    __requires(w == 1024);
    __requires(h == 1024);
    __requires(r ==   14);

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_y(&id[x], &od[x], w, h, r);
}
