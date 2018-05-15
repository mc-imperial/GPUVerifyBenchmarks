//pass
//--gridDim=[1322,1,1]     --blockDim=[256,1,1]

#include "common.h"

__global__ void getRepresentatives(const uint *successors,
                                   uint *representatives,
                                   uint verticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        uint successor = successors[tid];
        uint nextSuccessor = successors[successor];

        while (successor != nextSuccessor)
        {
            successor = nextSuccessor;
            nextSuccessor = successors[nextSuccessor];
        }

        representatives[tid] = successor;
    }
}
