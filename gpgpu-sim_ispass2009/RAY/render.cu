//pass
//--blockDim=32 --gridDim=1

#include "common.h"

__global__ void render(uint * result, Node * dnode, uint imageW, uint imageH, float pas, float df)
{
    __requires(imageW == 64);
    __requires(imageH == 64);
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint tid(__umul24(threadIdx.y, blockDim.x) + threadIdx.x);

	uint id(x + y * imageW);
	float4 pile[5];
	uint Obj, nRec(5), n(0);
	//__shared__ Node node[numObj];
	float prof, tmp;

	//if( tid < numObj ) node[tid] = cnode[tid];

	for( int i(0); i < nRec; ++i )
		pile[i] = make_float4(0.0f,0.0f,0.0f,1.0f);

	if( x < imageW && y < imageH )
	{
		prof = 10000.0f;
		result[id] = 0;
		float tPixel(2.0f/float(min(imageW,imageH)));
		float4 f(make_float4(0.0f,0.0f,0.0f,1.0f));
		matrice3x4 M(MView);
		Rayon R;
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
#ifdef DEVICE_EMU
//    printf("%d: R.A = %e %e %e\n", threadIdx.x, R.A.x, R.A.y, R.A.z);
//    printf("%d: R.u = %e %e %e\n", threadIdx.x, R.u.x, R.u.y, R.u.z);
#endif
#ifdef DEBUG_RT_CUDA
//    d_debug_float4[threadIdx.x*2+0].x= R.A.x;
//    d_debug_float4[threadIdx.x*2+0].y= R.A.y;
//    d_debug_float4[threadIdx.x*2+0].z= R.A.z;
//    d_debug_float4[threadIdx.x*2+1].x= R.u.x;
//    d_debug_float4[threadIdx.x*2+1].y= R.u.y;
//    d_debug_float4[threadIdx.x*2+1].z= R.u.z;
#endif
		__syncthreads();

		for( int i(0); i < nRec && n == i; i++ ) {

			for( int j(0); j < numObj; j++ ) {
				Node nod(cnode[j]);
				Sphere s(nod.s);
				float t;
				s.C.x += pas;
				if( nod.fg )
					t = intersectionPlan(R,s.C,s.C);
				else
					t = intersectionSphere(R,s.C,s.r);

				if( t > 0.0f && t < prof ) {
					prof = t;
					Obj = j;
				}
			}
#ifdef DEBUG_RT_CUDA
         //d_debug_float4[threadIdx.x*5+i].x= prof;
#endif
#ifdef DEVICE_EMU
//       printf("%d: i=%d, t=%e\n", threadIdx.x, i, prof);
#endif
			float t = prof;
			if( t > 0.0f && t < 10000.0f ) {
				n++;
				Node nod(cnode[Obj]);
				Sphere s(nod.s);
				s.C.x += pas;
				float4 color(make_float4(s.R,s.V,s.B,s.A));
				float3 P(R.A+R.u*t), L(normalize(make_float3(10.0f,10.0f,10.0f)-P)), V(normalize(R.A-P));
				float3 N(nod.fg?getNormaleP(P):getNormale(P,s.C));
				float3 Np(dot(V,N)<0.0f?(-1*N):N);
				pile[i] = 0.05f * color;
            #ifdef DEVICE_EMU
//          printf("%d: i=%d, pile[i] = %e %e %e %e\n", threadIdx.x, i, pile[i].x, pile[i].y, pile[i].z, pile[i].w);
//          printf("%d: i=%d, color = %e %e %e %e\n", threadIdx.x, i, color.x, color.y, color.z, color.w);
//          printf("%d: i=%d, P = %e %e %e\n", threadIdx.x, i, P.x, P.y, P.z);
//          printf("%d: i=%d, L = %e %e %e\n", threadIdx.x, i, L.x, L.y, L.z);
//          printf("%d: i=%d, V = %e %e %e\n", threadIdx.x, i, V.x, V.y, V.z);
//          printf("%d: i=%d, N = %e %e %e\n", threadIdx.x, i, N.x, N.y, N.z);
//          printf("%d: i=%d, Np = %e %e %e\n", threadIdx.x, i, Np.x, Np.y, Np.z);
//          printf("%d: i=%d, dot(Np,L) = %e\n", threadIdx.x, i, dot(Np,L));
            //printf("%d: i=%d, notShadowRay(cnode,P,L,pas) = %d\n", threadIdx.x, i, (int) notShadowRay(cnode,P,L,pas));

            #endif
            #ifdef DEBUG_RT_CUDA
            //d_debug_float4[threadIdx.x*16+i*3+0]= pile[i];
//          d_debug_float4[threadIdx.x*16+i*8+0]= color;
//          d_debug_float4[threadIdx.x*16+i*8+1].x= P.x;d_debug_float4[threadIdx.x*16+i*8+1].y= P.y;d_debug_float4[threadIdx.x*16+i*8+1].z= P.z;
//          d_debug_float4[threadIdx.x*16+i*8+2].x= L.x;d_debug_float4[threadIdx.x*16+i*8+2].y= L.y;d_debug_float4[threadIdx.x*16+i*8+2].z= L.z;
//          d_debug_float4[threadIdx.x*16+i*8+3].x= V.x;d_debug_float4[threadIdx.x*16+i*8+3].y= V.y;d_debug_float4[threadIdx.x*16+i*8+3].z= V.z;
//          d_debug_float4[threadIdx.x*16+i*8+4].x= N.x;d_debug_float4[threadIdx.x*16+i*8+4].y= N.y;d_debug_float4[threadIdx.x*16+i*8+4].z= N.z;
//          d_debug_float4[threadIdx.x*16+i*8+5].x= Np.x;d_debug_float4[threadIdx.x*16+i*8+5].y= Np.y;d_debug_float4[threadIdx.x*16+i*8+5].z= Np.z;
//          d_debug_float4[threadIdx.x*16+i*8+6].x= dot(Np,L);
            //d_debug_float4[threadIdx.x*16+i*8+7].x= (float) notShadowRay(cnode,P,L,pas);
            #endif
            #ifdef DEBUG_RT_CUDA
            if( dot(Np,L) > 0.0f && notShadowRay(d_debug_float4, d_debug_uint, i, cnode,P,L,pas) ) {
            #else
            if( dot(Np,L) > 0.0f && notShadowRay(cnode,P,L,pas) ) {
            #endif
               //float3 Ri(2.0f*Np*dot(Np,L) - L);
					float3 Ri(normalize(L+V));
					//Ri = (L+V)/normalize(L+V);
					pile[i] += 0.3f * color* (min(1.0f,dot(Np,L)));
               #ifdef DEVICE_EMU
//             printf("%d: i=%d, pile[i] = %e %e %e %e\n", threadIdx.x, i, pile[i].x, pile[i].y, pile[i].z, pile[i].w);
               #endif
               #ifdef DEBUG_RT_CUDA
               //d_debug_float4[threadIdx.x*16+i*3+1]= pile[i];
               #endif
               #ifdef FIXED_CONST_PARSE
					tmp = 0.8f * pow(max(0.0f,min(1.0f,dot(Np,Ri))),50.0f);
               #else
               tmp = 0.8f * float2int_pow50(max(0.0f,min(1.0f,dot(Np,Ri))));
               #endif
					pile[i].x += tmp;
					pile[i].y += tmp;
					pile[i].z += tmp;
               #ifdef DEVICE_EMU
//             printf("%d: i=%d, pile[i] = %e %e %e %e\n", threadIdx.x, i, pile[i].x, pile[i].y, pile[i].z, pile[i].w);
               #endif
               #ifdef DEBUG_RT_CUDA
               //d_debug_float4[threadIdx.x*16+i*3+2]= pile[i];
               #endif
				}

				R.u = 2.0f*N*dot(N,V) - V;
				R.u = normalize(R.u);
				R.A = P+R.u*0.0001f;
			}
			prof = 10000.0f;
		}
      #ifdef DEBUG_RT_CUDA
      /*d_debug_float4[threadIdx.x*5+0]= pile[0];
      d_debug_float4[threadIdx.x*5+1]= pile[1];
      d_debug_float4[threadIdx.x*5+2]= pile[2];
      d_debug_float4[threadIdx.x*5+3]= pile[3];
      d_debug_float4[threadIdx.x*5+4]= pile[4];*/
      #endif
#ifdef DEVICE_EMU
//    printf("%d: pile[0] = %e %e %e %e\n", threadIdx.x, pile[0].x, pile[0].y, pile[0].z, pile[0].w);
//    printf("%d: pile[1] = %e %e %e %e\n", threadIdx.x, pile[1].x, pile[1].y, pile[1].z, pile[1].w);
//    printf("%d: pile[2] = %e %e %e %e\n", threadIdx.x, pile[2].x, pile[2].y, pile[2].z, pile[2].w);
//    printf("%d: pile[3] = %e %e %e %e\n", threadIdx.x, pile[3].x, pile[3].y, pile[3].z, pile[3].w);
//    printf("%d: pile[4] = %e %e %e %e\n", threadIdx.x, pile[4].x, pile[4].y, pile[4].z, pile[4].w);
#endif
      for( int i(n-1); i > 0; i-- )
				pile[i-1] = pile[i-1] + 0.8f*pile[i];
#ifdef DEVICE_EMU
//    printf("%d: pile[0] = %e %e %e %e\n", threadIdx.x, pile[0].x, pile[0].y, pile[0].z, pile[0].w);
#endif
      result[id] += rgbaFloatToInt(pile[0]);
	}
}
