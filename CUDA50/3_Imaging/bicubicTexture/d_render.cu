//pass
//--gridDim=[32,32,1] --blockDim=[16,16,1]

#include "common.h"

__global__ void
d_render(uchar4 *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
    __requires(width == 512);
    __requires(height == 512);

    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = (x-cx)*scale+cx + tx;
    float v = (y-cy)*scale+cy + ty;

    if ((x < width) && (y < height))
    {
        // write output color
        float c = tex2D(tex, u, v);
        //float c = tex2DBilinear<uchar, float>(tex, u, v);
        //float c = tex2DBilinearGather<uchar, uchar4>(tex2, u, v, 0) / 255.0f;
        d_output[i] = make_uchar4(c * 0xff, c * 0xff, c * 0xff, 0);
    }
}
