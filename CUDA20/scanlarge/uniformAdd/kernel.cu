//pass
//--gridDim=128 --blockDim=128

#include <cuda.h>

__global__ void uniformAdd(float *g_data, 
                           float *uniforms, 
                           int n, 
                           int blockOffset, 
                           int baseIndex)
{
    __shared__ float uni[1];
    if (threadIdx.x == 0)
        uni[0] = uniforms[blockIdx.x + blockOffset];
         /* BUGINJECT: MUTATE_OFFSET, UP, ZERO */
    
    unsigned int address = blockIdx.x * (blockDim.x << 1) + baseIndex + threadIdx.x; 

    __syncthreads();
    
    // note two adds per thread
#ifdef MUTATION // couldn't apply mutation above; apply here instead
    g_data[0]                    += uni[0];
#else
    g_data[address]              += uni[0];
#endif
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni[0];
}

