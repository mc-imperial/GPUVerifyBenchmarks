//pass
//--blockDim=32 --gridDim=2

#include "../common.h"

__global__ void Pathcalc_Portfolio_KernelGPU(float *d_v, float *d_Lb)
{
  const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int threadN = blockDim.x * gridDim.x;

  int   i,path;
  float L[NN], L2[L2_SIZE], z[NN];
  float *L_b = L;
  
  /* Monte Carlo LIBOR path calculation*/

  for(path = tid; path < NPATH; path += threadN){
    // initialise the data for current thread
    for (i=0; i<N; i++) {
      // for real application, z should be randomly generated
      z[i] = 0.3;
      L[i] = 0.05;
    }
    path_calc_b1(L, z, L2);
    d_v[path] = portfolio_b(L,L_b);
    path_calc_b2(L_b, z, L2);
    d_Lb[path] = L_b[NN-1];
  }
}
