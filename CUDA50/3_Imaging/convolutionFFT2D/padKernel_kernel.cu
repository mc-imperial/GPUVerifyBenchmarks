//pass
//--gridDim=[1,1,1]        --blockDim=[32,8,1]

#include "common.h"

__global__ void padKernel_kernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
    __requires(fftH == 2048);
    __requires(fftW == 2048);

    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y < kernelH && x < kernelW)
    {
        int ky = y - kernelY;

        if (ky < 0)
        {
            ky += fftH;
        }

        int kx = x - kernelX;

        if (kx < 0)
        {
            kx += fftW;
        }

        d_Dst[ky * fftW + kx] = LOAD_FLOAT(y * kernelW + x);
    }
}
