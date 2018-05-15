//pass
//--gridDim=[192,128,1]    --blockDim=[16,12,1]

#include "common.h"

__global__ void convolutionColumnsKernel(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    __requires(imageW == 3072);
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionColumn<2 *KERNEL_RADIUS>(x, y);
#else

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D(texSrc, x, y + (float)k) * c_Kernel[KERNEL_RADIUS - k];
    }

#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}
