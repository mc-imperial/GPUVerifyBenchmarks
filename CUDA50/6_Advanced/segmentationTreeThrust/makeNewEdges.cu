//pass
//--gridDim=[11377,1,1]    --blockDim=[256,1,1]

#include "common.h"

__global__ void makeNewEdges(const uint *survivedEdgesIDs,
                             const uint *verticesMapping,
                             const uint *edges,
                             const float *weights,
                             uint *newEdges,
                             float *newWeights,
                             uint edgesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < edgesCount)
    {
        uint edgeID = survivedEdgesIDs[tid];
        uint oldEdge = edges[edgeID];

        newEdges[tid] = verticesMapping[oldEdge];
        newWeights[tid] = weights[edgeID];
    }
}
