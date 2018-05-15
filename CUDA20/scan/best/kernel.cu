//pass
//--blockDim=[32,1] --gridDim=[1,1]

#include <cuda.h>

#define N 32

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
//#define ZERO_BANK_CONFLICTS 

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
// #define CONFLICT_FREE_OFFSET(index) (index)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   cutilBankChecker(temp, index)
#else
#define TEMP(index)   temp[index]
#endif

///////////////////////////////////////////////////////////////////////////////
// Work-efficient compute implementation of scan, one thread per 2 elements
// Work-efficient: O(log(n)) steps, and O(n) adds.
// Also shared storage efficient: Uses n + n/NUM_BANKS shared memory -- no ping-ponging
// Also avoids most bank conflicts using single-element offsets every NUM_BANKS elements.
//
// In addition, If ZERO_BANK_CONFLICTS is defined, uses 
//     n + n/NUM_BANKS + n/(NUM_BANKS*NUM_BANKS) 
// shared memory. If ZERO_BANK_CONFLICTS is defined, avoids ALL bank conflicts using 
// single-element offsets every NUM_BANKS elements, plus additional single-element offsets 
// after every NUM_BANKS^2 elements.
//
// Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
// and Their Applications", or Prins and Chatterjee PRAM course notes:
// http://www.cs.unc.edu/~prins/Classes/203/Handouts/pram.pdf
// 
// This work-efficient version is based on the algorithm presented in Guy Blelloch's
// Excellent paper "Prefix sums and their applications".
// http://www-2.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html
//
// Pro: Work Efficient, very few bank conflicts (or zero if ZERO_BANK_CONFLICTS is defined)
// Con: More instructions to compute bank-conflict-free shared memory addressing,
// and slightly more shared memory storage used.
//
// @param g_odata  output data in global memory
// @param g_idata  input data in global memory
// @param n        input number of elements to scan from input data
__global__ void scanBestKernel(float *g_odata, float *g_idata, int n)
{
    __requires(n == blockDim.x*2);
    __requires(__is_pow2(n));

    // Dynamically allocated shared memory for scan kernels
    /*extern*/ __shared__  float temp[N*2];

    int thid = threadIdx.x;

#ifdef NORENAME
    int ai = thid;
    int bi = thid + (n >> 1);

    // compute spacing to avoid bank conflicts
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    TEMP(ai + bankOffsetA) = g_idata[ai]; 
    TEMP(bi + bankOffsetB) = g_idata[bi]; 
#else
    int ai_outer = thid;
    int bi_outer = thid + (n >> 1);

    // compute spacing to avoid bank conflicts
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai_outer);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi_outer);

    // Cache the computational window in shared memory
    TEMP(ai_outer + bankOffsetA) = g_idata[ai_outer]; 
#ifdef MUTATION
    TEMP(0) = g_idata[bi_outer]; 
#else
    TEMP(bi_outer + bankOffsetB) = g_idata[bi_outer]; 
#endif
      /* BUGINJECT: MUTATE_OFFSET, UP, ZERO */
#endif

    __syncthreads();

    int offset = 1;

    // build the sum in place up the tree
    for (int d = n >> 1;
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

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            TEMP(bi) += TEMP(ai);
        }

    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
        int index = n - 1;
        index += CONFLICT_FREE_OFFSET(index);
        TEMP(index) = 0;
    }

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset /= 2;

        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t  = TEMP(ai);
            TEMP(ai) = TEMP(bi);
            TEMP(bi) += t;
        }
    }

    __syncthreads();

    // write results to global memory
#ifdef NORENAME
    g_odata[ai] = TEMP(ai + bankOffsetA); 
    g_odata[bi] = TEMP(bi + bankOffsetB); 
#else
    g_odata[ai_outer] = TEMP(ai_outer + bankOffsetA); 
    g_odata[bi_outer] = TEMP(bi_outer + bankOffsetB); 
#endif

    
}
