//pass
//--blockDim=[16,16] --gridDim=[32,32]

#include <cuda.h>

//////////////////////////////////////////////////////////////////////////////
//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved
//////////////////////////////////////////////////////////////////////////////

#define X_DIMENSION 0
#define Y_DIMENSION 1

// Original kernels are templated.  We will check the float case.
#define _type float

//----------------------------------------------------------------------------
// File: BitonicSort.cpp
// 
// Implements Bitonic sort in C++ AMP
// Supports only int, unsigned, long and unsigned long
//----------------------------------------------------------------------------

#define BITONIC_TILE_SIZE          512
// Should be a square matrix
#define NUM_ELEMENTS                (BITONIC_TILE_SIZE * BITONIC_TILE_SIZE) 
#define MATRIX_WIDTH                BITONIC_TILE_SIZE
#define MATRIX_HEIGHT               BITONIC_TILE_SIZE
// Should be divisible by MATRIX_WIDTH and MATRIX_HEIGHT
// else parallel_for_each will crash
#define TRANSPOSE_TILE_SIZE        16

//----------------------------------------------------------------------------
// Kernel implements 2D matrix transpose
//----------------------------------------------------------------------------
__global__ void transpose_kernel(_type* data_in, _type* data_out, unsigned width, unsigned height)
{
  __requires(width == 512 /*MATRIX_WIDTH*/);

  __shared__ _type transpose_shared_data[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE];

  transpose_shared_data[threadIdx.y][threadIdx.x] = data_in[(blockDim.y*blockIdx.y + threadIdx.y)*width + (blockDim.x*blockIdx.x + threadIdx.x)];

#ifndef MUTATION
   /* BUGINJECT: REMOVE_BARRIER, DOWN */
  __syncthreads();
#endif

  data_out[(blockDim.x*blockIdx.x + threadIdx.x)*width + (blockDim.y*blockIdx.y + threadIdx.y)] = transpose_shared_data[threadIdx.y][threadIdx.x];
}
