//pass
//--gridDim=[1322,1,1]     --blockDim=[256,1,1]

#include "common.h"

__global__ void removeCycles(uint *successors,
                             uint verticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        uint successor = successors[tid];
        uint nextSuccessor = successors[successor];

        if (tid == nextSuccessor)
        {
            if (tid < successor)
            {
                successors[tid] = tid;
            }
            else
            {
                successors[successor] = successor;
            }
        }
    }
}
