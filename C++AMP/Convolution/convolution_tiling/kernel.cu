//pass
//--blockDim=[1,128] --gridDim=[512,6]

#include <cuda.h>

//////////////////////////////////////////////////////////////////////////////
//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// File: Convolution.cpp
// 
// Implement C++ AMP based simple and tiled version of Convolution filter used in 
// image processing.
//----------------------------------------------------------------------------

#define DEFAULT_WIDTH   512
#define DEFAULT_HEIGHT  512
// TILE_SIZE should be multiple of both DEFAULT_WIDTH and DEFAULT_HEIGHT
#define TILE_SIZE		128

#define width DEFAULT_WIDTH
#define height DEFAULT_HEIGHT

#define clamp(a, b, c) ((a) < (b) ? (b) : ((a) > (c) ? (c) : (a)))

#define dim_to_convolve y

#define radius 7

//----------------------------------------------------------------------------
// Tile implementation of convolution filter along different dimension
//----------------------------------------------------------------------------
__global__ void convolution_tiling(const float* img, const float* filter, float* result)
{

    __shared__ float local_buf[TILE_SIZE];
    
    int idx_convolve = (blockIdx.dim_to_convolve)*(TILE_SIZE - 2 * radius) + (int)(threadIdx.dim_to_convolve) - radius;
    int max_idx_convolve = height;
    float sum = 0.0f;

    int a_idxY = blockIdx.y;
    int a_idxX = blockIdx.x;

    a_idxY = clamp(idx_convolve, 0, max_idx_convolve-1);
    if (idx_convolve < (max_idx_convolve + radius))
    {
        local_buf[threadIdx.dim_to_convolve] = img[a_idxY*width + a_idxX];
    }

#ifndef MUTATION
     /* BUGINJECT: REMOVE_BARRIER, DOWN */
    __syncthreads();
#endif

    if ((int)(threadIdx.dim_to_convolve) >= radius && (int)(threadIdx.dim_to_convolve) < (TILE_SIZE - radius) && idx_convolve < max_idx_convolve)
    {
        for (int k = -radius; k <= radius; k++)
        {
            int k_idx = k + radius;
            sum += local_buf[threadIdx.dim_to_convolve + k]*filter[k_idx];
        }
        result[a_idxY*width + a_idxX] = sum;
    }
}
