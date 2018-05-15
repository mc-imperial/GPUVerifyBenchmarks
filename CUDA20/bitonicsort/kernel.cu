//pass
//--blockDim=[32,1] --gridDim=[1,1]

#include <cuda.h>

#define NUM 32

__global__ void BitonicKernel(int * values)
{
  __shared__ int shared[NUM];

  unsigned int tid = threadIdx.x;

  // Copy input to shared mem.
  shared[tid] = values[tid];

#ifdef MUTATION
  if (threadIdx.x == 0) {
#endif
  __syncthreads();
#ifdef MUTATION
   /* BUGINJECT: NON_UNIFORM_CONTROL_FLOW, UP */
  }
#endif

  // Parallel bitonic sort.
  for (unsigned int k = 2;
       k <= NUM; k *= 2)
  {
    // Bitonic merge:
    for (unsigned int j = k / 2;
       j>0; j /= 2)
    {
      unsigned int ixj = tid ^ j;

      if (ixj > tid)
      {
        if ((tid & k) == 0)
        {
          if (shared[tid] > shared[ixj])
          {
            unsigned int tmp = shared[tid];
            shared[tid] = shared[ixj];
            shared[ixj] = shared[tid];
          }
        }
        else
        {
          if (shared[tid] < shared[ixj])
          {
            unsigned int tmp = shared[tid];
            shared[tid] = shared[ixj];
            shared[ixj] = shared[tid];
          }
        }
      }

      __syncthreads();
    }
  }

  // Write result.
  values[tid] = shared[tid];
}
