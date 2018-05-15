//pass
//--blockDim=32 --gridDim=2

#include "../common.h"

__global__ void MaxwellsGPU_RK_Kernel3D(int Ntotal, float *g_resQ, float *g_rhsQ, float *g_Q, float fa, float fb, float fdt){
  
  int n = blockIdx.x * blockDim.x + threadIdx.x;
    
  if(n<Ntotal){
    float rhs = g_rhsQ[n];
    float res = g_resQ[n];
    res = fa*res + fdt*rhs;
    
    g_resQ[n] = res;
    g_Q[n]    += fb*res;
  }

} 
