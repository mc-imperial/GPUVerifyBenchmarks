//pass
//--blockDim=[128,128] --gridDim=[4,4]

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

#define radius 7

#define width DEFAULT_WIDTH
#define height DEFAULT_HEIGHT

#define clamp(a, b, c) ((a) < (b) ? (b) : ((a) > (c) ? (c) : (a)))

//----------------------------------------------------------------------------
// Simple implementation of convolution filter along different dimension
//----------------------------------------------------------------------------
static __attribute__((always_inline)) __device__ float convolution_dim_simple(const float* img, const float* filter)
{
    float sum = 0.0f;
    for (int k = -radius; k <= radius; k++)
    {
        int dim = clamp((blockDim.y*blockIdx.y + threadIdx.y) + k, 0, height-1);

        int aIdxX = (blockDim.x*blockIdx.x + threadIdx.x);
        int aIdxY = dim;

        int kidx = k + radius;
        sum += img[aIdxY*width + aIdxX]*filter[kidx];
    }
    return sum;
}

//----------------------------------------------------------------------------
// Simple implementation of convolution separable filter 
//----------------------------------------------------------------------------
__global__ void convolution_simple(float* v_img, float* v_filter, float* v_result)
{
  v_result[(blockDim.y*blockIdx.y + threadIdx.y)*width + (blockDim.x*blockIdx.x + threadIdx.x)] = convolution_dim_simple(v_img, v_filter);
#ifdef MUTATION
  v_result[(blockDim.y*blockIdx.y + threadIdx.y)*width + (blockDim.x*blockIdx.x + threadIdx.x) + 1] = v_result[(blockDim.y*blockIdx.y + threadIdx.y)*width + (blockDim.x*blockIdx.x + threadIdx.x) + 1];
   /* BUGINJECT: ADD_ACCESS, UP */
#endif
}
