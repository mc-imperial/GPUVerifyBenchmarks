//pass
//--gridDim=16 --blockDim=64

#include "common.h"
  
__global__ void
d_boxfilter_rgba_x(unsigned int *od, int w, int h, int r)
{
    __requires(w == 1024);
    __requires(h == 1024);

    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;

    // as long as address is always less than height, we do work
    if (y < h)
    {
        float4 t = make_float4(0.0f);

        for (int x = -r; x <= r; x++)
        {
            t += tex2D(rgbaTex, x, y);
        }

        od[y * w] = rgbaFloatToInt(t * scale);

        for (int x = 1;
             x < w; x++)
        {
            t += tex2D(rgbaTex, x + r, y);
            t -= tex2D(rgbaTex, x - r - 1, y);
            od[y * w + x] = rgbaFloatToInt(t * scale);
        }
    }
}
