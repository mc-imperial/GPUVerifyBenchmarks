//pass
//--gridDim=[64,64,1]      --blockDim=[16,16,1]

//REQUIRES: cudaExtent
//REQUIRES: SURFACE

#include "common.h"

__global__ void
d_preintegrate(int layer, float steps, cudaExtent extent)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= extent.width || y >= extent.height)
    {
        return;
    }

    float sx = float(x)/float(extent.width);
    float sy = float(y)/float(extent.height);

    float smax = max(sx,sy);
    float smin = min(sx,sy);

    float4 iv;

    if (x != y)
    {
        // assumes square textures!
        float fracc = smax - smin;
        fracc = 1.0 /(fracc*steps);

        float4 intmax = tex1D(transferIntegrateTex,smax);
        float4 intmin = tex1D(transferIntegrateTex,smin);
        iv.x = (intmax.x - intmin.x)*fracc;
        iv.y = (intmax.y - intmin.y)*fracc;
        iv.z = (intmax.z - intmin.z)*fracc;
        //iv.w = (intmax.w - intmin.w)*fracc;
        iv.w   = (1.0 - exp(-(intmax.w - intmin.w) * fracc));
    }
    else
    {
        float4 sample = tex1D(transferTex,smin);
        iv.x = sample.x;
        iv.y = sample.y;
        iv.z = sample.z;
        //iv.w = sample.w;
        iv.w   = (1.0 - exp(-sample.w));
    }

    iv.x =  __saturatef(iv.x);
    iv.y =  __saturatef(iv.y);
    iv.z =  __saturatef(iv.z);
    iv.w =  __saturatef(iv.w);

    // surface writes need byte offsets for x!
    surf2DLayeredwrite(iv,transferLayerPreintSurf, x * sizeof(float4), y, layer);
}
