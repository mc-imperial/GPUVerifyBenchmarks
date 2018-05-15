//pass
//--blockDim=32 --gridDim=2

#include "../common.h"
__global__ void partial_get_kernel3d(int Ntotal, int *g_index, float *g_partQ){
  
  int n = blockIdx.x * blockDim.x + threadIdx.x;
    
  if(n<Ntotal)
    g_partQ[n] = tex1Dfetch(t_Q, g_index[n]);
  
} 
