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


#define num_vertices (1 << 6)

// State of connection
#define UNCONNECTED 0
#define DIRECTLY_CONNECTED 1
#define INDIRECTLY_CONNECTED 2

//----------------------------------------------------------------------------
// Stage1 - determine connectivity between vertexs' within a TILE - primary
//----------------------------------------------------------------------------

__global__ void transitive_closure_stage1_kernel(unsigned int* graph, int passnum)
{
    
    // Load primary block into shared memory (primary_block_buffer)
    __shared__ unsigned int primary_block_buffer[TILE_SIZE][TILE_SIZE];

    // TODO: check that in OpenCL the order is 0=x, 1=y, 2=z (in AMP it is reversed)
    int idxY = passnum * TILE_SIZE + threadIdx.y;
    int idxX = passnum * TILE_SIZE + threadIdx.x;

    primary_block_buffer[threadIdx.y][threadIdx.x] = _2D_ACCESS(graph, idxY, idxX, num_vertices);

#ifndef MUTATION
     /* BUGINJECT: REMOVE_BARRIER, DOWN */
    __syncthreads();
#endif

    // Now perform the actual Floyd-Warshall algorithm on this block
    for (unsigned int k = 0;
         k < TILE_SIZE; ++k)
    {
        if ( primary_block_buffer[threadIdx.y][threadIdx.x] == UNCONNECTED)
        {
            if ( (primary_block_buffer[threadIdx.y][k] != UNCONNECTED) && (primary_block_buffer[k][threadIdx.x] != UNCONNECTED) )
            {
                primary_block_buffer[threadIdx.y][threadIdx.x] = passnum*TILE_SIZE + k + INDIRECTLY_CONNECTED;
            }
        }

        __syncthreads();
    }

    _2D_ACCESS(graph, idxY, idxX, num_vertices) = primary_block_buffer[threadIdx.y][threadIdx.x];
}

