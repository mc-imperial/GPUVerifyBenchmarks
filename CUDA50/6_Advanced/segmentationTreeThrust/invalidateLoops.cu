//pass
//--gridDim=[11377,1,1]    --blockDim=[256,1,1]

#include "common.h"

__global__ void invalidateLoops(const uint *startpoints,
                                const uint *verticesMapping,
                                uint *edges,
                                uint edgesCount)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < edgesCount)
    {
        uint startpoint = startpoints[tid];
        uint &endpoint = edges[tid];

        uint newStartpoint = verticesMapping[startpoint];
        uint newEndpoint = verticesMapping[endpoint];

        if (newStartpoint == newEndpoint)
        {
            endpoint = UINT_MAX;
        }
    }
}
