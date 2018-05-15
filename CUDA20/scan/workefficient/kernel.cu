//pass
//--blockDim=[32,1] --gridDim=[1,1]

#include <cuda.h>

#define N 32

///////////////////////////////////////////////////////////////////////////////
//! Work-efficient compute implementation of scan, one thread per 2 elements
//! Work-efficient: O(log(n)) steps, and O(n) adds.
//! Also shared storage efficient: Uses n elements in shared mem -- no ping-ponging
//! Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
//! and Their Applications", or Prins and Chatterjee PRAM course notes:
//! http://www.cs.unc.edu/~prins/Classes/203/Handouts/pram.pdf
//!
//! Pro: Work Efficient
//! Con: Shared memory bank conflicts due to the addressing used.
//
//! @param g_odata  output data in global memory
//! @param g_idata  input data in global memory
//! @param n        input number of elements to scan from input data
///////////////////////////////////////////////////////////////////////////////

__global__ void scan_workefficient_kernel (float *g_odata, float *g_idata, int n)
{
    __requires(n == blockDim.x*2);
    __requires(__is_pow2(n));

    // Dynamically allocated shared memory for scan kernels
    /*extern*/ __shared__  float temp[N*2];

    int thid = threadIdx.x;

    int offset = 1;

    // Cache the computational window in shared memory
    temp[2*thid]   = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];

    // build the sum in place up the tree
    for (int d = n>>1;
      __invariant(__implies((d == 0) & __write(temp), thid == 0)),
      __invariant(__implies((d == 0) & __read(temp), thid == 0)),
        d > 0; d >>= 1)
    {

      __syncthreads();

      offset *= 2;

      if (thid < d)
      {
        int ai = offset/2*(2*thid+1)-1;
        int bi = offset/2*(2*thid+2)-1;

        temp[bi] = 1;
        temp[bi] += temp[ai];
      }
    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
        temp[n - 1] = 0;
    }

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
      __syncthreads();

      offset >>= 1;

      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;

        float t = temp[ai];
        temp[ai]  = temp[bi];
        temp[bi] += t;
      }
    }

#ifndef MUTATION
     /* BUGINJECT: REMOVE_BARRIER, DOWN */
    __syncthreads();
#endif

    // write results to global memory
    g_odata[2*thid]   = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}
