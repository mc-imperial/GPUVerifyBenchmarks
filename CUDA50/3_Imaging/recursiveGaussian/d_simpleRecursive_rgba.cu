//pass
//--gridDim=8 --blockDim=64

#include "common.h"

__global__ void
d_simpleRecursive_rgba(uint *id, uint *od, int w, int h, float a)
{
    __requires(w == 512);
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x >= w) return;

    id += x;    // advance pointers to correct column
    od += x;

    // forward pass
    float4 yp = rgbaIntToFloat(*id);  // previous output

    for (int y = 0;
         __global_invariant((__ptr_offset_bytes(od)/sizeof(uint))%w == x),
         y < h; y++)
    {
        float4 xc = rgbaIntToFloat(*id);
        float4 yc = xc + a*(yp - xc);   // simple lerp between current and previous value
        *od = rgbaFloatToInt(yc);
        id += w;
        od += w;    // move to next row
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    yp = rgbaIntToFloat(*id);

    for (int y = h-1;
         __global_invariant((__ptr_offset_bytes(od)/sizeof(uint))%w == x),
         __global_invariant(__write_implies(od, (__write_offset_bytes(od)/sizeof(uint))%w == x)),
         __global_invariant(__read_implies(od, (__read_offset_bytes(od)/sizeof(uint))%w == x)),
         y >= 0; y--)
    {
        float4 xc = rgbaIntToFloat(*id);
        float4 yc = xc + a*(yp - xc);
        *od = rgbaFloatToInt((rgbaIntToFloat(*od) + yc)*0.5f);
        id -= w;
        od -= w;  // move to previous row
        yp = yc;
    }
}
