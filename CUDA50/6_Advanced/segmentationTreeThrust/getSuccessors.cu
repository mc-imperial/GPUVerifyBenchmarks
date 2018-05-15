//pass
//--gridDim=[1322,1,1]     --blockDim=[256,1,1]

#include "common.h"

__global__ void getSuccessors(const uint *verticesOffsets,
                              const uint *minScannedEdges,
                              uint *successors,
                              uint verticesCount,
                              uint edgesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount)
    {
        uint successorPos = (tid < verticesCount - 1) ?
                            (verticesOffsets[tid + 1] - 1) :
                            (edgesCount - 1);

        successors[tid] = minScannedEdges[successorPos];
    }
}
