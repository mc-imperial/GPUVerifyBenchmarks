//pass
//--gridDim=1 --blockDim=[9,9]

#include "common.h"

__global__ void
addForces_k(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch)
{
    __requires(dx == 512);
    __requires(dy == 512);
    __requires(spx == 1);
    __requires(spy == 1);
    __requires(r == 4);
    __requires(pitch == 4096);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    cData *fj = (cData *)((char *)v + (ty + spy) * pitch) + tx + spx;

    cData vterm = *fj;
    tx -= r;
    ty -= r;
    float s = 1.f / (1.f + tx*tx*tx*tx + ty*ty*ty*ty);
    vterm.x += s * fx;
    vterm.y += s * fy;
    *fj = vterm;
}
