//pass
//--gridDim=[4800,1,1]     --blockDim=[256,1,1]

#include "common.h"

__global__ void markSegments(const uint *verticesOffsets,
                             uint *flags,
                             uint verticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        flags[verticesOffsets[tid]] = 1;
    }
}
