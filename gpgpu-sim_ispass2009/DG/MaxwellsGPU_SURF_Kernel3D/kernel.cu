//pass
//--blockDim=32 --gridDim=2

#include "../common.h"

__global__ void MaxwellsGPU_SURF_Kernel3D(float *g_Q, float *g_rhsQ){

  __device__ __shared__ float s_fluxQ[p_Nfields*p_Nfp*p_Nfaces];

  const int n = threadIdx.x;
  const int k = blockIdx.x;
  int m;

  /* grab surface nodes and store flux in shared memory */
  if(n< (p_Nfp*p_Nfaces) ){
    /* coalesced reads (maybe) */
    m = 7*(k*p_Nfp*p_Nfaces)+n;
    const  int idM   = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
           int idP   = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float Fsc = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float Bsc = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float nx  = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float ny  = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float nz  = tex1Dfetch(t_surfinfo, m);

    /* check if idP<0  */
    double dHx, dHy, dHz, dEx, dEy, dEz;
    if(idP<0){
      idP = p_Nfields*(-1-idP);
      
      dHx = Fsc*(tex1Dfetch(t_partQ, idP+0) - tex1Dfetch(t_Q, idM+0*BSIZE));
      dHy = Fsc*(tex1Dfetch(t_partQ, idP+1) - tex1Dfetch(t_Q, idM+1*BSIZE));
      dHz = Fsc*(tex1Dfetch(t_partQ, idP+2) - tex1Dfetch(t_Q, idM+2*BSIZE));
      
      dEx = Fsc*(tex1Dfetch(t_partQ, idP+3) - tex1Dfetch(t_Q, idM+3*BSIZE));
      dEy = Fsc*(tex1Dfetch(t_partQ, idP+4) - tex1Dfetch(t_Q, idM+4*BSIZE));
      dEz = Fsc*(tex1Dfetch(t_partQ, idP+5) - tex1Dfetch(t_Q, idM+5*BSIZE));
    }
    else{
      dHx = Fsc*(tex1Dfetch(t_Q, idP+0*BSIZE) - tex1Dfetch(t_Q, idM+0*BSIZE));
      dHy = Fsc*(tex1Dfetch(t_Q, idP+1*BSIZE) - tex1Dfetch(t_Q, idM+1*BSIZE));
      dHz = Fsc*(tex1Dfetch(t_Q, idP+2*BSIZE) - tex1Dfetch(t_Q, idM+2*BSIZE));
      
      dEx = Fsc*(Bsc*tex1Dfetch(t_Q, idP+3*BSIZE) - tex1Dfetch(t_Q, idM+3*BSIZE));
      dEy = Fsc*(Bsc*tex1Dfetch(t_Q, idP+4*BSIZE) - tex1Dfetch(t_Q, idM+4*BSIZE));
      dEz = Fsc*(Bsc*tex1Dfetch(t_Q, idP+5*BSIZE) - tex1Dfetch(t_Q, idM+5*BSIZE));
    }

    const double ndotdH = nx*dHx + ny*dHy + nz*dHz;
    const double ndotdE = nx*dEx + ny*dEy + nz*dEz;

    m = n;
    s_fluxQ[m] = -ny*dEz + nz*dEy + dHx - ndotdH*nx; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] = -nz*dEx + nx*dEz + dHy - ndotdH*ny; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] = -nx*dEy + ny*dEx + dHz - ndotdH*nz; m += p_Nfp*p_Nfaces;

    s_fluxQ[m] =  ny*dHz - nz*dHy + dEx - ndotdE*nx; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] =  nz*dHx - nx*dHz + dEy - ndotdE*ny; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] =  nx*dHy - ny*dHx + dEz - ndotdE*nz; 
  }

  /* make sure all element data points are cached */
  __syncthreads();

  if(n< (p_Np))
  {
    float rhsHx = 0, rhsHy = 0, rhsHz = 0;
    float rhsEx = 0, rhsEy = 0, rhsEz = 0;
    
    int sk = n;
    /* can manually unroll to 4 because there are 4 faces */
    for(m=0;p_Nfaces*p_Nfp-m;){
      const float4 L = tex1Dfetch(t_LIFT, sk); sk+=p_Np;

      /* broadcast */
      int sk1 = m;
      rhsHx += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

    }
    
    m = n+p_Nfields*k*BSIZE;
    g_rhsQ[m] += rhsHx; m += BSIZE;
    g_rhsQ[m] += rhsHy; m += BSIZE;
    g_rhsQ[m] += rhsHz; m += BSIZE;
    g_rhsQ[m] += rhsEx; m += BSIZE;
    g_rhsQ[m] += rhsEy; m += BSIZE;
    g_rhsQ[m] += rhsEz; m += BSIZE;

  }
}
