//pass
//--gridDim=1024 --blockDim=1024

#include <cuda.h>

__global__ void square_array(float* dataView)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  dataView[idx] = dataView[idx] * dataView[idx];
#ifdef MUTATION
  dataView[idx+1] = dataView[idx+1];
#endif
   /* BUGINJECT: ADD_ACCESS, UP */
}
