//pass
//--blockDim=[64,64] --gridDim=[4,4]

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

#define X_DIMENSION 0
#define Y_DIMENSION 1


//----------------------------------------------------------------------------
// Implement simple matrix multiplication on GPU using C++ AMP
// M, N and W are sizes of matrix
// input matrix - va is of size (M * N), vb is (N * W) 
// output matrix - vresult (M * W)
//----------------------------------------------------------------------------
__global__ void mxm_amp_simple(const _type * va, const _type * vb, _type * vresult)
{
    // Compute - outer 2 for loops of CPU are replaced by a parallel_for_each
        {
            _type result = 0.0f;

            for(int i = 0; i < N; ++i)
            {
                int idx_a_X = i;
                int idx_a_Y = blockIdx.y*blockDim.y + threadIdx.y;

                int idx_b_X = blockIdx.x*blockDim.x + threadIdx.x;
                int idx_b_Y = i; 

                result += va[idx_a_Y*M + idx_a_X] * vb[idx_b_Y*N + idx_b_X];
            }

            vresult[(blockIdx.y*blockDim.y + threadIdx.y)*M + (blockIdx.x*blockDim.x + threadIdx.x)] = result;
#ifdef MUTATION
            vresult[(blockIdx.y*blockDim.y + threadIdx.y)*M + (blockIdx.x*blockDim.x + threadIdx.x) + 1] = vresult[(blockIdx.y*blockDim.y + threadIdx.y)*M + (blockIdx.x*blockDim.x + threadIdx.x) + 1];
             /* BUGINJECT: ADD_ACCESS, UP */
#endif
        }
}
