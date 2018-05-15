//pass
//--gridDim=[1322,1,1]     --blockDim=[256,1,1]

#include "common.h"

__global__ void getVerticesMapping(const uint *clusteredVerticesIDs,
                                   const uint *newVerticesIDs,
                                   uint *verticesMapping,
                                   uint verticesCount)
{
    __requires(
     clusteredVerticesIDs[blockIdx.x * blockDim.x + threadIdx.x] !=
     clusteredVerticesIDs[__other_int(blockIdx.x * blockDim.x + threadIdx.x)]);

    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    __requires(clusteredVerticesIDs[tid] != clusteredVerticesIDs[__other_int(tid)]);

    if (tid < verticesCount)
    {
        uint vertexID = clusteredVerticesIDs[tid];
        verticesMapping[vertexID] = newVerticesIDs[tid];
    }
}
