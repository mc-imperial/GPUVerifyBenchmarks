//pass
//--blockDim=[1024,1] --gridDim=[4,1]

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
// Write MT_RNG_COUNT vertical lanes of n_per_RNG random numbers to random_nums.
// For coalesced global writes MT_RNG_COUNT should be a multiple of hardware scehduling unit size.
// Hardware scheduling unit is called warp or wave or wavefront
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small n_per_RNG supply dedicated (local) seed to each twister.
// The local seeds, in their turn, can be extracted from global seed
// by means of any simple random number generator, like LCG.
////////////////////////////////////////////////////////////////////////////////
__global__ void rand_MT_kernel(
			   float* random_nums, 
			   const unsigned int matrix_a, 
			   const unsigned int mask_b, const unsigned int mask_c, 
			   const unsigned int seed, const int n_per_RNG)
{
    int state_1;
    int state_M;
    unsigned int mti, mti_M, x;
    unsigned int mti_1, mt[MT_NN];

    //Bit-vector Mersenne Twister parameters are in matrix_a, mask_b, mask_c, seed
    //Initialize current state
    mt[0] = seed;
    for(int state = 1; state < MT_NN; state++)
        mt[state] = (1812433253U * (mt[state - 1] ^ (mt[state - 1] >> 30)) + state) & MT_WMASK;

    mti_1 = mt[0];
    for(int out = 0, state = 0; 
          out < n_per_RNG; out++) 
	{
        state_1 = state + 1;
        state_M = state + MT_MM;
        if (state_1 >= MT_NN) state_1 -= MT_NN;
        if (state_M >= MT_NN) state_M -= MT_NN;
        mti  = mti_1;
        mti_1 = mt[state_1];
        mti_M = mt[state_M];

        x    = (mti & MT_UMASK) | (mti_1 & MT_LMASK);
        x    =  mti_M ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);
        mt[state] = x;
        state = state_1;

        //Tempering transformation
        x ^= (x >> MT_SHIFT0);
        x ^= (x << MT_SHIFTB) & mask_b;
        x ^= (x << MT_SHIFTC) & mask_c;
        x ^= (x >> MT_SHIFT1);

        
        //Convert to (0, 1) float and write to global memory
		// Using UINT max, to convert a uniform number in uint range to a uniform range over [-1 ... 1] 
        random_nums[out*MT_RNG_COUNT + (blockIdx.x * blockDim.x + threadIdx.x)] = ((float)x + 1.0f) / 4294967296.0f;
#ifdef MUTATION
        random_nums[out*MT_RNG_COUNT + (blockIdx.x * blockDim.x + threadIdx.x) + 1] = random_nums[out*MT_RNG_COUNT + (blockIdx.x * blockDim.x + threadIdx.x) + 1];
         /* BUGINJECT: ADD_ACCESS, UP */
#endif
    }
}

