//pass
//--blockDim=32 --gridDim=2

#include "../common.h"

__global__ void Pathcalc_Portfolio_KernelGPU2(float *d_v)
{
  const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int threadN = blockDim.x * gridDim.x;

  int   i, path;
  float L[NN], z[NN];
  
  /* Monte Carlo LIBOR path calculation*/

  for(path = tid; path < NPATH; path += threadN){
    // initialise the data for current thread
    for (i=0; i<N; i++) {
      // for real application, z should be randomly generated
      z[i] = 0.3;
      L[i] = 0.05;
    }	   
    path_calc(L, z);
    d_v[path] = portfolio(L);
  }
}
