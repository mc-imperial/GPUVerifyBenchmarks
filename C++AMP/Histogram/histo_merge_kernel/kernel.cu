//pass
//--blockDim=256 --gridDim=256

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
// File: Histogram.cpp
// 
// Implements histogram in C++ AMP
// Refer README.txt
//----------------------------------------------------------------------------

#define histogram_bin_count 256 /* Bin count */

#define log2_thread_size 5U
#define thread_count 8 /* number of partial histogram per tile */

#define histogram256_tile_size (thread_count * (1U << log2_thread_size))
#define histogram256_tile_static_memory (thread_count * histogram_bin_count)

#define merge_tile_size histogram_bin_count /* Partial result Merge size */
#define partial_histogram256_count (thread_count * (1U << log2_thread_size))

// This function aggregates partial results
__global__ void histo_merge_kernel(unsigned int* partial_result, unsigned int* histogram_amp)
{

        {
            unsigned sum = 0;
            for (unsigned i = threadIdx.x;
                   i < partial_histogram256_count * histogram_bin_count; i += merge_tile_size)
            {
                sum += partial_result[blockIdx.x + i * histogram_bin_count];
            }

            __shared__ unsigned s_data[merge_tile_size];
            s_data[threadIdx.x] = sum;

            // parallel reduce within a tile
            for (int stride = merge_tile_size / 2;
                     stride > 0; stride >>= 1)
            {
#ifndef MUTATION
                 /* BUGINJECT: REMOVE_BARRIER, DOWN */
                __syncthreads();
#endif
                if (threadIdx.x < stride)
                {
                    s_data[threadIdx.x] += s_data[threadIdx.x + stride];
                }
            }

            // tile sum is updated to result array by zero-th thread
            if (threadIdx.x == 0)
            {
                histogram_amp[blockIdx.x] = s_data[0];
            }
        }
}
