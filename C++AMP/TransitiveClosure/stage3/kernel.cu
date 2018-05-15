//pass
//--blockDim=[8,8] --gridDim=[1,1]

#include <cuda.h>

#define _2D_ACCESS(A, y, x, X_DIM) A[(y)*(X_DIM)+(x)]

//////////////////////////////////////////////////////////////////////////////
//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// File: TransitiveClosure.cpp
//
// Contains the implementation of algorithms which explores connectivity between 
// nodes in a graph and determine shortest path.
// This is based on paper http://www.seas.upenn.edu/~kiderj/research/papers/APSP-gh08-fin-T.pdf
//----------------------------------------------------------------------------

// Defines to help with AMP->OpenCL translation
#define X_DIMENSION 0
#define Y_DIMENSION 1

// Constants - specifies tile size
#define TILE_SIZE (1 << 3)

// State of connection
#define UNCONNECTED 0
#define DIRECTLY_CONNECTED 1
#define INDIRECTLY_CONNECTED 2


#define num_vertices (1 << 6)

//----------------------------------------------------------------------------
// Stage3 - determine connectivity between vertexs' between 3 TILE 
// 1. primary block, 2. block made of row af current and column of primary 
// 3. block made of column of current and row of primary
//----------------------------------------------------------------------------
__global__ void transitive_closure_stage3_kernel(unsigned int* graph, int passnum)
{
    unsigned int group_id0, group_id1;
    if (blockIdx.y < passnum)
    {
        group_id0 = blockIdx.y;
    }
    else
    {
        group_id0 = blockIdx.y + 1;
    }

    if (blockIdx.x < passnum)
    {
        group_id1 = blockIdx.x;
    }
    else
    {
        group_id1 = blockIdx.x + 1;
    }

    // Load block with same row as current block and same column as primary block into shared memory (shBuffer1)
    __shared__ unsigned int shbuffer1[TILE_SIZE][TILE_SIZE];

    int idxY = group_id0 * TILE_SIZE + threadIdx.y;
    int idxX = passnum * TILE_SIZE + threadIdx.x;
    shbuffer1[threadIdx.y][threadIdx.x] = _2D_ACCESS(graph, idxY, idxX, num_vertices);

    // Load block with same column as current block and same row as primary block into shared memory (shBuffer2)
    __shared__ unsigned int shBuffer2[TILE_SIZE][TILE_SIZE];
    idxY = passnum * TILE_SIZE + threadIdx.y;
    idxX = group_id1 * TILE_SIZE + threadIdx.x;
    shBuffer2[threadIdx.y][threadIdx.x] = _2D_ACCESS(graph, idxY, idxX, num_vertices);

    //  Load the current block into shared memory (shbuffer3)
    __shared__ unsigned int curr_block_buffer[TILE_SIZE][TILE_SIZE];
    idxY = group_id0 * TILE_SIZE + threadIdx.y;
    idxX = group_id1 * TILE_SIZE + threadIdx.x;
    curr_block_buffer[threadIdx.y][threadIdx.x] = _2D_ACCESS(graph, idxY, idxX, num_vertices);

#ifndef MUTATION
     /* BUGINJECT: REMOVE_BARRIER, DOWN */
    __syncthreads();
#endif

    // Now perform the actual Floyd-Warshall algorithm on this block
    for (unsigned int k = 0; k < TILE_SIZE; ++k)
    {
        if ( curr_block_buffer[threadIdx.y][threadIdx.x] == UNCONNECTED)
        {
            if ( (shbuffer1[threadIdx.y][k] != UNCONNECTED) && (shBuffer2[k][threadIdx.x] != UNCONNECTED) )
            {
                curr_block_buffer[threadIdx.y][threadIdx.x] = passnum*TILE_SIZE + k + INDIRECTLY_CONNECTED;
            }
        }

        __syncthreads();
    }

    _2D_ACCESS(graph, idxY, idxX, num_vertices) = curr_block_buffer[threadIdx.y][threadIdx.x];
}
