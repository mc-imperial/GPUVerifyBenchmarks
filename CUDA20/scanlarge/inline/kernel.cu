//pass
//--blockDim=[32,1] --gridDim=[1,1]

#include <cuda.h>

#define N 32

// Define this to more rigorously avoid bank conflicts, 
// even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance 
// is lower with ZERO_BANK_CONFLICTS enabled.  It is provided
// as an example.
//#define ZERO_BANK_CONFLICTS 

// 16 banks on G80
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

__global__ void k(float *g_odata, 
                        const float *g_idata, 
                        float *g_blockSums, 
                        int n, 
                        int blockIndex, 
                        int baseIndex,
                        int storeSum, int isNP2)
{
  int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
  /*extern*/ __shared__ float s_data[N*2];

  // ------------------------------------------------------------------------
  // loadSharedChunkFromMem()
  // ------------------------------------------------------------------------
  baseIndex = (baseIndex == 0) ?  blockIdx.x * (blockDim.x << 1) : baseIndex;
  int thid = threadIdx.x;
  mem_ai = baseIndex + threadIdx.x;
  mem_bi = mem_ai + blockDim.x;

  ai = thid;
  bi = thid + blockDim.x;

  // compute spacing to avoid bank conflicts
  bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  // Cache the computational window in shared memory
  // pad values beyond n with zeros

  s_data[ai + bankOffsetA] = g_idata[mem_ai];

  if (isNP2 != 0) // compile-time decision
  {
    s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0;
  }
  else
  {
    s_data[bi + bankOffsetB] = g_idata[mem_bi];
  }

  // ------------------------------------------------------------------------
  // prescanBlock()
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // -- buildSum()
  // ------------------------------------------------------------------------
  unsigned int stride = 1;
  // build the sum in place up the tree
  for (int d = blockDim.x;
    __invariant(__implies((d == 0) & __write(s_data), thid == 0)),
    __invariant(__implies((d == 0) & __read(s_data), thid == 0)),
          d > 0; d >>= 1)
  {
    __syncthreads();

    stride *= 2;

    if (thid < d)
    {
      int i_1  = stride * thid;
      int ai_1 = i_1 + stride/2 - 1;
      int bi_1 = ai_1 + stride/2;

      ai_1 += CONFLICT_FREE_OFFSET(ai_1);
      bi_1 += CONFLICT_FREE_OFFSET(bi_1);

#ifdef MUTATION
      s_data[0] += s_data[ai_1];
#else
      s_data[bi_1] += s_data[ai_1];
#endif
       /* BUGINJECT: MUTATE_OFFSET, UP, ZERO */
    }
  }

  // ------------------------------------------------------------------------
  // -- clearLastElement()
  // ------------------------------------------------------------------------
  blockIndex = (blockIndex == 0) ? blockIdx.x : blockIndex;
  if (threadIdx.x == 0)
  {
    int index = (blockDim.x << 1) - 1;
    index += CONFLICT_FREE_OFFSET(index);

    if (storeSum != 0) // compile-time decision
    {
      // write this block's total sum to the corresponding index in the blockSums array
      g_blockSums[blockIndex] = s_data[index];
    }

    // zero the last element in the scan so it will propagate back to the front
    s_data[index] = 0;
  }

  // ------------------------------------------------------------------------
  // -- scanRootToLeaves()
  // ------------------------------------------------------------------------
  for (int d = 1; d <= blockDim.x; d *= 2)
  {
    stride >>= 1;

    __syncthreads();

    if (thid < d)
    {
      int i_2  =  2 * stride * thid;
      int ai_2 = i_2 + stride - 1;
      int bi_2 = ai_2 + stride;
      ai_2 += CONFLICT_FREE_OFFSET(ai_2);
      bi_2 += CONFLICT_FREE_OFFSET(bi_2);

      float t      = s_data[ai_2];
      s_data[ai_2] = s_data[bi_2];
      s_data[bi_2] += t; 
    }
  }

  // ------------------------------------------------------------------------
  // storeSharedChunkToMem()
  // ------------------------------------------------------------------------
  __syncthreads();

  // write results to global memory
  g_odata[mem_ai] = s_data[ai + bankOffsetA]; 
  if (isNP2 != 0) // compile-time decision
  {
    if (bi < n)
      g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
  }
  else
  {
    g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
  }
}
