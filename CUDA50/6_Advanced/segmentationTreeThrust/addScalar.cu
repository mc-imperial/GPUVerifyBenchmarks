//pass
//--gridDim=[11377,1,1]    --blockDim=[256,1,1]

#include "common.h"

__global__ void addScalar(uint *array, int scalar, uint size)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        array[tid] += scalar;
    }
}
