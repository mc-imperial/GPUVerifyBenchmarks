//pass
//--gridDim=16 --blockDim=64

#include "common.h"
  
__global__ void
d_boxfilter_x_tex(float *od, int w, int h, int r)
{
    __requires(w == 1024);
    __requires(h == 1024);

    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int x =- r; x <= r; x++)
    {
        t += tex2D(tex, x, y);
    }

    od[y * w] = t * scale;

    for (int x = 1;
         x < w; x++)
    {
        t += tex2D(tex, x + r, y);
        t -= tex2D(tex, x - r - 1, y);
        od[y * w + x] = t * scale;
    }
}
