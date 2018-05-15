#define p_N 6
#define p_Nfp     ((p_N+1)*(p_N+2)/2)
#define p_Np      ((p_N+1)*(p_N+2)*(p_N+3)/6)
#define p_Nfields 6
#define p_Nfaces  4
#define BSIZE   (16*((p_Np+15)/16))

texture<float4, 1, cudaReadModeElementType> t_LIFT;
texture<float4, 1, cudaReadModeElementType> t_DrDsDt;
texture<float, 1, cudaReadModeElementType> t_Dr;
texture<float, 1, cudaReadModeElementType> t_Ds;
texture<float, 1, cudaReadModeElementType> t_Dt;
texture<float, 1, cudaReadModeElementType> t_vgeo;
texture<float4, 1, cudaReadModeElementType> t_vgeo4;
texture<float, 1, cudaReadModeElementType> t_Q;
texture<float, 1, cudaReadModeElementType> t_partQ;
texture<float, 1, cudaReadModeElementType> t_surfinfo;
