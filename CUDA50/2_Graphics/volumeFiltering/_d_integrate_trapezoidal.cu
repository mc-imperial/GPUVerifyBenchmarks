//pass
//--gridDim=[32,1,1]       --blockDim=[32,1,1]

//REQUIRES: cudaExtent
//REQUIRES: SURFACE

#include "common.h"

__global__ void
d_integrate_trapezoidal(cudaExtent extent)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;

    // for higher speed could use hierarchical approach for sum
    if (x >= extent.width)
    {
        return;
    }

    float stepsize = 1.0/float(extent.width-1);
    float to = float(x) * stepsize;

    float4 outclr = make_float4(0,0,0,0);
    float incr = stepsize;

    float4 lastval = tex1D(transferTex,0);

    float cur = incr;

    while (cur < to + incr * 0.5)
    {
        float4 val = tex1D(transferTex,cur);
        float4 trapezoid = (lastval+val)/2.0f;
        lastval = val;

        outclr += trapezoid;
        cur += incr;
    }

    // surface writes need byte offsets for x!
    surf1Dwrite(outclr,transferIntegrateSurf,x * sizeof(float4));
}
