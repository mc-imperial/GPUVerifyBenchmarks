//pass
//--blockDim=256 --gridDim=512

//ALTERNATIVELY:
//--blockDim=16 --gridDim=8 -DSMALL


#include <cuda.h>

//#define SMALL

#define fast_min(x, y) ((x) < (y) ? (x) : (y))

//////////////////////////////////////////////////////////////////////////////
//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// File: BinomialOptions.cpp
// 
// Implement GPU based binomial option pricing. Verify correctness with CPU 
// implementation
//----------------------------------------------------------------------------

#ifdef SMALL

// Date set - small and normal
// small problem size
#define  MAX_OPTIONS    (32)
#define  NUM_STEPS      (64)
#define  TIME_STEPS     (2)
#define  CACHE_DELTA    (2 * TIME_STEPS)
#define  CACHE_SIZE     (16)
#define  CACHE_STEP     (CACHE_SIZE - CACHE_DELTA)

#else

// normal problem size
#define  MAX_OPTIONS    (512)
#define  NUM_STEPS      (2048)
#define  TIME_STEPS     (16)
#define  CACHE_DELTA    (2 * TIME_STEPS)
#define  CACHE_SIZE     (256)
#define  CACHE_STEP     (CACHE_SIZE - CACHE_DELTA)

#endif

#if NUM_STEPS % CACHE_DELTA
    #error Bad constants
#endif


//----------------------------------------------------------------------------
// GPU implementation - Call value at period t : V(t) = S(t) - X
//----------------------------------------------------------------------------
static __attribute__((always_inline)) __device__ float expiry_call_value(float s, float x, float vdt, int t)
{
    float d = s * exp(vdt * (2.0f * t - NUM_STEPS)) - x;
    return (d > 0) ? d : 0;
}

//----------------------------------------------------------------------------
// GPU implementation of binomial options tree walking to calculate option pricing
// Refer README.txt for more details on algorithm
//----------------------------------------------------------------------------
// Using #ifdef to workaround an exception on Window 7 and Debug build
// Runtime throw's an exception:
//		ID3D11DeviceContext::Dispatch: The Shader Resource View in slot 0 of the Compute 
//  Shader unit is a Structured Buffer while the shader expects a typed Buffer.  This 
// mismatch is invalid if the shader actually uses the view (e.g. it is not skipped due to shader code branching).
// This issue will be fixed in next release.
__global__ void binomial_options_kernel(
                   const float* s, const float* x, 
                   const float* vdt, const float* pu_by_df, 
                   const float* pd_by_df,
                   float* call_value, 
                   float* call_buffer) 
{
  int tile_idx = blockIdx.x;
  int local_idx = threadIdx.x;

  __shared__ float call_a[CACHE_SIZE+1];
  __shared__ float call_b[CACHE_SIZE+1];

  //Global memory frame for current option (thread group)
  int tid = local_idx;

  // CACHE_SIZE number of thread are operating, hence steping by CACHE_SIZE
  // below for loop is similar to first inner loop of binomial_options_cpu
  //Compute values at expiry date
  for(int index = tid; index <= NUM_STEPS; index += CACHE_SIZE)
  {
    int idxA = tile_idx * (NUM_STEPS + 16) + (index);
    call_buffer[idxA] = expiry_call_value(s[tile_idx], x[tile_idx], vdt[tile_idx], index);
  }

  // Walk down binomial tree - equivalent to 2nd inner loop of binomial_options_cpu
  //                              Additional boundary checking 
  // So double-buffer and synchronize to avoid read-after-write hazards.
  for(int i = NUM_STEPS; i > 0; i -= CACHE_DELTA)
  {

    for(int c_base = 0; c_base < i; c_base += CACHE_STEP)
    {
      // Start and end positions within shared memory cache
      int c_start = fast_min(CACHE_SIZE - 1, i - c_base);
      int c_end   = c_start - CACHE_DELTA;

      // Read data(with apron) to shared memory
#ifndef MUTATION
       /* BUGINJECT: REMOVE_BARRIER, DOWN */
      __syncthreads();
#endif
      if(tid <= c_start)
      {
        int idxB = tile_idx * (NUM_STEPS + 16) + (c_base + tid);
        call_a[tid] = call_buffer[idxB];
      }

      // Calculations within shared memory
      for(int k = c_start - 1; 
        k >= c_end;)
      {
        // Compute discounted expected value
        __syncthreads();
        call_b[tid] = pu_by_df[tile_idx] * call_a[tid + 1] + pd_by_df[tile_idx] * call_a[tid];
        k--;

        // Compute discounted expected value
        __syncthreads();
        call_a[tid] = pu_by_df[tile_idx] * call_b[tid + 1] + pd_by_df[tile_idx] * call_b[tid];
        k--;
      }

      // Flush shared memory cache
      __syncthreads();
      if(tid <= c_end)
      {
        int idxC = tile_idx * (NUM_STEPS + 16) + (c_base + tid);
        call_buffer[idxC] = call_a[tid];
      }
    }
  }

  // Write the value at the top of the tree to destination buffer
  if (tid == 0) 
    call_value[tile_idx] = call_a[0];
}
