//pass
//--gridDim=8 --blockDim=64

#include "common.h"

__global__ void
d_recursiveGaussian_rgba(uint *id, uint *od, int w, int h, float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn)
{
    __requires(w == 512);
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x >= w) return;

    id += x;    // advance pointers to correct column
    od += x;

    // forward pass
    float4 xp = make_float4(0.0f);  // previous input
    float4 yp = make_float4(0.0f);  // previous output
    float4 yb = make_float4(0.0f);  // previous output by 2
#if CLAMP_TO_EDGE
    xp = rgbaIntToFloat(*id);
    yb = coefp*xp;
    yp = yb;
#endif

    for (int y = 0;
         __global_invariant((__ptr_offset_bytes(od)/sizeof(uint))%w == x),
         y < h; y++)
    {
        float4 xc = rgbaIntToFloat(*id);
        float4 yc = a0*xc + a1*xp - b1*yp - b2*yb;
        *od = rgbaFloatToInt(yc);
        id += w;
        od += w;    // move to next row
        xp = xc;
        yb = yp;
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    float4 xn = make_float4(0.0f);
    float4 xa = make_float4(0.0f);
    float4 yn = make_float4(0.0f);
    float4 ya = make_float4(0.0f);
#if CLAMP_TO_EDGE
    xn = xa = rgbaIntToFloat(*id);
    yn = coefn*xn;
    ya = yn;
#endif

    for (int y = h-1;
         __global_invariant((__ptr_offset_bytes(od)/sizeof(uint))%w == x),
         __global_invariant(__write_implies(od, (__write_offset_bytes(od)/sizeof(uint))%w == x)),
         __global_invariant(__read_implies(od, (__read_offset_bytes(od)/sizeof(uint))%w == x)),
         y >= 0; y--)
    {
        float4 xc = rgbaIntToFloat(*id);
        float4 yc = a2*xn + a3*xa - b1*yn - b2*ya;
        xa = xn;
        xn = xc;
        ya = yn;
        yn = yc;
        *od = rgbaFloatToInt(rgbaIntToFloat(*od) + yc);
        id -= w;
        od -= w;  // move to previous row
    }
}
