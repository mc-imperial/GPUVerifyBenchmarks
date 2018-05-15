//pass
//--blockDim=1024 --gridDim=4

#include <cuda.h>

//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) Microsoft Corporation. All rights reserved
//// This software contains source code provided by NVIDIA Corporation.
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// File: MersenneTwister.cpp
// 
// This sample implements Mersenne Twister random number generator 
// and Cartesian Box-Muller transformation on the GPU.
//----------------------------------------------------------------------------

#define   MT_RNG_COUNT 4096
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18

////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of n_per_RNG uniformly distributed 
// random samples, produced by rand_MT_amp(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// n_per_RNG must be even.
////////////////////////////////////////////////////////////////////////////////
static __attribute__((always_inline)) __device__ void box_muller_transform(float* u1, float* u2)
{
    float r = sqrt(-2.0f * log(*u1));
    float phi = 2.0f * 3.14159265358979f * (*u2);
    *u1 = r * cos(phi);
    *u2 = r * sin(phi);
}

__global__ void box_muller_kernel(float* random_nums, float* normalized_random_nums, int n_per_RNG)
{
    int gid = (blockIdx.x*blockDim.x + threadIdx.x);

    for(int out = 0;
                  out < n_per_RNG; out += 2) 
	{
		float f0 = random_nums[out * MT_RNG_COUNT + gid];
		float f1 = random_nums[(out + 1) * MT_RNG_COUNT + gid];
                box_muller_transform(&f0, &f1);
                normalized_random_nums[out * MT_RNG_COUNT + gid] = f0;
                normalized_random_nums[(out + 1) * MT_RNG_COUNT + gid] = f1;
#ifdef MUTATION
    normalized_random_nums[out * MT_RNG_COUNT + gid + 1] = normalized_random_nums[out * MT_RNG_COUNT + gid + 1];
                 /* BUGINJECT: ADD_ACCESS, UP */
#endif
    }
}

