//pass
//--gridDim=[8,8,1] --blockDim=[64,4,1]

#include "common.h"

texture<float2, 2> texref;

__global__ void
advectVelocity_k(cData *v, float *vx, float *vy,
                 int dx, int pdx, int dy, float dt, int lb)
{
    __requires(dx == 512);
    __requires(dy == 512);
    __requires(lb == 16);
    __requires(pdx == 514);

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    cData vterm, ploc;
    float vxterm, vyterm;

    // gtidx is the domain location in x for this thread
    if (gtidx < dx)
    {
        for (p = 0;
             __global_invariant(__write_implies(vx, ((__write_offset_bytes(vx)/sizeof(int))%pdx == gtidx))),
             __global_invariant(__write_implies(vy, ((__write_offset_bytes(vy)/sizeof(int))%pdx == gtidx))),
             p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fj = fi * pdx + gtidx;
                vterm = tex2D(texref, (float)gtidx, (float)fi);
                ploc.x = (gtidx + 0.5f) - (dt * vterm.x * dx);
                ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
                vterm = tex2D(texref, ploc.x, ploc.y);
                vxterm = vterm.x;
                vyterm = vterm.y;
                vx[fj] = vxterm;
                vy[fj] = vyterm;
            }
        }
    }
}
