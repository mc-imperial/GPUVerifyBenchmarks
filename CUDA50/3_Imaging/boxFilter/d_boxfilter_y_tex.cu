//pass
//--gridDim=16 --blockDim=64

#include "common.h"
  
__global__ void
d_boxfilter_y_tex(float *od, int w, int h, int r)
{
    __requires(w == 1024);
    __requires(h == 1024);

    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int y = -r; y <= r; y++)
    {
        t += tex2D(tex, x, y);
    }

    od[x] = t * scale;

    for (int y = 1; y < h; y++)
    {
        t += tex2D(tex, x, y + r);
        t -= tex2D(tex, x, y - r - 1);
        od[y * w + x] = t * scale;
    }
}
