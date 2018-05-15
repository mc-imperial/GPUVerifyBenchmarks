//pass
//--gridDim=[64,256,1]     --blockDim=[32,8,1]

#include "common.h"

__global__ void padDataClampToBorder_kernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
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
    const int borderH = dataH + kernelY;
    const int borderW = dataW + kernelX;

    if (y < fftH && x < fftW)
    {
        int dy, dx;

        if (y < dataH)
        {
            dy = y;
        }

        if (x < dataW)
        {
            dx = x;
        }

        if (y >= dataH && y < borderH)
        {
            dy = dataH - 1;
        }

        if (x >= dataW && x < borderW)
        {
            dx = dataW - 1;
        }

        if (y >= borderH)
        {
            dy = 0;
        }

        if (x >= borderW)
        {
            dx = 0;
        }

        d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
    }
}
