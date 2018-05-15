//pass
//--gridDim=[8,8,1] --blockDim=[64,4,1]

#include "common.h"
  
__global__ void
updateVelocity_k(cData *v, float *vx, float *vy,
                 int dx, int pdx, int dy, int lb, size_t pitch)
{
    __requires(dx == 512);
    __requires(dy == 512);
    __requires(lb == 16);
    __requires(pitch == 4096);

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    float vxterm, vyterm;
    cData nvterm;

    // gtidx is the domain location in x for this thread
    if (gtidx < dx)
    {
        for (p = 0;
           __global_invariant(__write_implies(v, __write_offset_bytes(v)/pitch/lb%blockDim.y == threadIdx.y)),
           __global_invariant(__write_implies(v, __write_offset_bytes(v)/pitch/lb/blockDim.y == blockIdx.y)),
             p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fjr = fi * pdx + gtidx;
                vxterm = vx[fjr];
                vyterm = vy[fjr];

                // Normalize the result of the inverse FFT
                float scale = 1.f / (dx * dy);
                nvterm.x = vxterm * scale;
                nvterm.y = vyterm * scale;

                cData *fj = (cData *)((char *)v + fi * pitch) + gtidx;
                *fj = nvterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}
