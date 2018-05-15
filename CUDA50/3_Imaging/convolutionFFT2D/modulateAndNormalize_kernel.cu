//pass
//--gridDim=8320             --blockDim=256

#include "common.h"

__global__ void modulateAndNormalize_kernel(
    fComplex *d_Dst,
    fComplex *d_Src,
    int dataSize,
    float c
)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= dataSize)
    {
        return;
    }

    fComplex a = d_Src[i];
    fComplex b = d_Dst[i];

    mulAndScale(a, b, c);

    d_Dst[i] = a;
}
