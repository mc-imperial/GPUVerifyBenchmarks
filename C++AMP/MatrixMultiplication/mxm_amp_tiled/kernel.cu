//pass
//--blockDim=[16,16] --gridDim=[16,16]

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
// File: Matrixmult.cpp
// 
// Implement GPU based matrix multiplication
//----------------------------------------------------------------------------

#define _type float

#define M 256
#define N 256
#define W 256

#define tile_size 16

#define X_DIMENSION 0
#define Y_DIMENSION 1

//----------------------------------------------------------------------------
// Implement tiled version of matrix multiplication
// M, N and W are sizes of matrix
// input matrix - va is of size (M * N), vb is (N * W) 
// output matrix - vresult (M * W)
//----------------------------------------------------------------------------
__global__ void mxm_amp_tiled(const _type * va, const _type * vb, _type * vresult)
{

	{
		__shared__ _type localB[tile_size][tile_size];
		__shared__ _type localA[tile_size][tile_size];

		_type temp_c = 0;

		int localIdxX = threadIdx.x;
                int localIdxY = threadIdx.y;
		int globalIdxX = blockIdx.x*blockDim.x + threadIdx.x;
                int globalIdxY = blockIdx.y*blockDim.y + threadIdx.y;
  
		for (int i = 0;
                       i < N; i += tile_size)
		{

			localA[localIdxY][localIdxX] = va[globalIdxY*M + i + localIdxX];
			localB[localIdxY][localIdxX] = vb[(i + localIdxY)*N + globalIdxX];
#ifndef MUTATION
    /* BUGINJECT: REMOVE_BARRIER, DOWN */
			__syncthreads();
#endif
        
			for (unsigned int k = 0; k < tile_size; k++)
			{
				temp_c += localA[localIdxY][k] * localB[k][localIdxX];
			}

                        __syncthreads();       
		}

		vresult[globalIdxY*M + globalIdxX] = temp_c;
	}
}
