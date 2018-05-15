//pass
//--blockDim=[32,1] --gridDim=[1,1]

#include <cuda.h>

#define N 32

///////////////////////////////////////////////////////////////////////////////
//! Naive compute implementation of scan, one thread per element
//! Not work efficient: log(n) steps, but n * (log(n) - 1) adds.
//! Not shared storage efficient either -- this requires ping-ponging
//! arrays in shared memory due to hazards so 2 * n storage space.
//!
//! Pro: Simple
//! Con: Not work efficient
//!
//! @param g_odata  output data in global memory
//! @param g_idata  input data in global memory
//! @param n        input number of elements to scan from input data
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(float *g_odata, float *g_idata, int n)
{
    __requires(n == blockDim.x); //< n is a pow2 and equal to blockDim.x

    // REVISIT: removed extern
    // REVISIT: give temp static size
    // Dynamically allocated shared memory for scan kernels
    /*extern*/__shared__  float temp[N*2];

    int thid = threadIdx.x;

    int pout = 0;
    int pin = 1;

    // Cache the computational window in shared memory
    temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;

    for (int offset = 1;
         offset < n; offset *= 2)
    {

        pout = 1 - pout;
        pin  = 1 - pout;

#ifndef MUTATION
        /* BUGINJECT: REMOVE_BARRIER, DOWN */
        __syncthreads();
#endif
        temp[pout*n+thid] = temp[pin*n+thid];

        if (thid >= offset) {
             temp[pout*n+thid] += temp[pin*n+thid - offset];
        }
    }

    __syncthreads();

    g_odata[thid] = temp[pout*n+thid];
}
