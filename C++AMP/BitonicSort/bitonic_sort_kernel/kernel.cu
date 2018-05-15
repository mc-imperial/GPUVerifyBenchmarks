//pass
//--blockDim=512 --gridDim=512

#include <cuda.h>

//////////////////////////////////////////////////////////////////////////////
//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved
//////////////////////////////////////////////////////////////////////////////

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
// Kernel implements partial sorting on accelerator, BITONIC_TILE_SIZE at a time
//----------------------------------------------------------------------------
__global__ void bitonic_sort_kernel(_type* data, unsigned ulevel, unsigned ulevelmask)
{
    __shared__ _type sh_data[BITONIC_TILE_SIZE];

    int local_idx = threadIdx.x;
    int global_idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Cooperatively load data - each thread will load data from global memory
    // into tile_static
    sh_data[local_idx] = data[global_idx];

    // Wait till all threads have loaded their portion of data
#ifndef MUTATION
     /* BUGINJECT: REMOVE_BARRIER, DOWN */
    __syncthreads();
#endif
    
    // Sort data in tile_static memory
    for (unsigned int j = ulevel >> 1 ;
        j > 0 ; j >>= 1)
    {
        _type result = ((sh_data[local_idx & ~j] <= sh_data[local_idx | j]) == (bool)(ulevelmask & global_idx)) ? sh_data[local_idx ^ j] : sh_data[local_idx];
        __syncthreads();
        sh_data[local_idx] = result;
        __syncthreads();
    }
    
    // Store shared data
    data[global_idx] = sh_data[local_idx];
}

