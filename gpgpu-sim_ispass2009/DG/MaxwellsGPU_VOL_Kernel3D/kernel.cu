//pass
//--blockDim=32 --gridDim=2

#include "../common.h"

__global__ void MaxwellsGPU_VOL_Kernel3D(float *g_rhsQ){

  /* fastest */
  __device__ __shared__ float s_Q[p_Nfields*BSIZE];
  __device__ __shared__ float s_facs[12];

  const int n = threadIdx.x;
  const int k = blockIdx.x;
  
  /* "coalesced"  */
  int m = n+k*p_Nfields*BSIZE;
  int id = n;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); 

  if(p_Np<12 && n==0)
    for(m=0;m<12;++m)
      s_facs[m] = tex1Dfetch(t_vgeo, 12*k+m);
  else if(n<12 && p_Np>=12)
    s_facs[n] = tex1Dfetch(t_vgeo, 12*k+n);

  __syncthreads();

  float dHxdr=0,dHxds=0,dHxdt=0;
  float dHydr=0,dHyds=0,dHydt=0;
  float dHzdr=0,dHzds=0,dHzdt=0;
  float dExdr=0,dExds=0,dExdt=0;
  float dEydr=0,dEyds=0,dEydt=0;
  float dEzdr=0,dEzds=0,dEzdt=0;
  float Q;

  for(m=0;p_Np-m;){
    float4 D = tex1Dfetch(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;
#if ( (p_Np) % 2 )==0
    D = tex1Dfetch(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;

#if ( (p_Np)%3 )==0
    D = tex1Dfetch(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;
#endif
#endif
  }
  
  const float drdx= s_facs[0];
  const float drdy= s_facs[1];
  const float drdz= s_facs[2];
  const float dsdx= s_facs[4];
  const float dsdy= s_facs[5];
  const float dsdz= s_facs[6];
  const float dtdx= s_facs[8];
  const float dtdy= s_facs[9];
  const float dtdz= s_facs[10];
  
  m = n+p_Nfields*BSIZE*k;

  g_rhsQ[m] = -(drdy*dEzdr+dsdy*dEzds+dtdy*dEzdt - drdz*dEydr-dsdz*dEyds-dtdz*dEydt); m += BSIZE;
  g_rhsQ[m] = -(drdz*dExdr+dsdz*dExds+dtdz*dExdt - drdx*dEzdr-dsdx*dEzds-dtdx*dEzdt); m += BSIZE;
  g_rhsQ[m] = -(drdx*dEydr+dsdx*dEyds+dtdx*dEydt - drdy*dExdr-dsdy*dExds-dtdy*dExdt); m += BSIZE;
  g_rhsQ[m] =  (drdy*dHzdr+dsdy*dHzds+dtdy*dHzdt - drdz*dHydr-dsdz*dHyds-dtdz*dHydt); m += BSIZE;
  g_rhsQ[m] =  (drdz*dHxdr+dsdz*dHxds+dtdz*dHxdt - drdx*dHzdr-dsdx*dHzds-dtdx*dHzdt); m += BSIZE;
  g_rhsQ[m] =  (drdx*dHydr+dsdx*dHyds+dtdx*dHydt - drdy*dHxdr-dsdy*dHxds-dtdy*dHxdt); 
}
