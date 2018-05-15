//pass
//--gridDim=[11377,1,1]    --blockDim=[256,1,1]

#include "common.h"

__global__ void calculateEdgesInfo(const uint *startpoints,
                                   const uint *verticesMapping,
                                   const uint *edges,
                                   const float *weights,
                                   uint *newStartpoints,
                                   uint *survivedEdgesIDs,
                                   uint edgesCount,
                                   uint newVerticesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < edgesCount)
    {
        uint startpoint = startpoints[tid];
        uint endpoint = edges[tid];

        newStartpoints[tid] = endpoint < UINT_MAX ?
                              verticesMapping[startpoint] :
                              newVerticesCount + verticesMapping[startpoint];

        survivedEdgesIDs[tid] = endpoint < UINT_MAX ?
                                tid :
                                UINT_MAX;
    }
}
