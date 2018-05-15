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
// Stage2 - determine connectivity between vertexs' between 2 TILE - primary 
// and current - current is along row or column of primary
//----------------------------------------------------------------------------
__global__ void transitive_closure_stage2_kernel(unsigned int* graph, int passnum)
{
  // Load primary block into shared memory (primary_block_buffer)
  __shared__ unsigned int primary_block_buffer[TILE_SIZE][TILE_SIZE];
  int idxY = passnum * TILE_SIZE + threadIdx.y;
  int idxX = passnum * TILE_SIZE + threadIdx.x;

  primary_block_buffer[threadIdx.y][threadIdx.x] = _2D_ACCESS(graph, idxY, idxX, num_vertices);

  // Load the current block into shared memory (curr_block_buffer)
  __shared__ unsigned int curr_block_buffer[TILE_SIZE][TILE_SIZE];
  unsigned int group_id0, group_id1;
  if (blockIdx.y == 0)
  {
    group_id0 = passnum;
    if (blockIdx.x < passnum)
    {
      group_id1 = blockIdx.x;
    }
    else
    {
      group_id1 = blockIdx.x + 1;
    }
  }
  else
  {
    group_id1 = passnum;
    if (blockIdx.x < passnum)
    {
      group_id0 = blockIdx.x;
    }
    else
    {
      group_id0 = blockIdx.x + 1;
    }
  }

  idxY = group_id0 * TILE_SIZE + threadIdx.y;
  idxX = group_id1 * TILE_SIZE + threadIdx.x;
  curr_block_buffer[threadIdx.y][threadIdx.x] = _2D_ACCESS(graph, idxY, idxX, num_vertices);

#ifndef MUTATION
   /* BUGINJECT: REMOVE_BARRIER, DOWN */
  __syncthreads();
#endif

  // Now perform the actual Floyd-Warshall algorithm on this block
  for (unsigned int k = 0;
                k < TILE_SIZE; ++k)
  {
    
    if ( curr_block_buffer[threadIdx.y][threadIdx.x] == UNCONNECTED)
    {
      if (blockIdx.y == 0)
      {
        if ( (primary_block_buffer[threadIdx.y][k] != UNCONNECTED) && (curr_block_buffer[k][threadIdx.x] != UNCONNECTED) )
        {
          curr_block_buffer[threadIdx.y][threadIdx.x] = passnum*TILE_SIZE + k + INDIRECTLY_CONNECTED;
        }
      }
      else
      {
        if ( (curr_block_buffer[threadIdx.y][k] != UNCONNECTED) && (primary_block_buffer[k][threadIdx.x] != UNCONNECTED) )
        {
          curr_block_buffer[threadIdx.y][threadIdx.x] = passnum*TILE_SIZE + k + INDIRECTLY_CONNECTED;
        }
      }
    }

    __syncthreads();
  }

  _2D_ACCESS(graph, idxY, idxX, num_vertices) = curr_block_buffer[threadIdx.y][threadIdx.x];
}
