//pass
//--blockDim=[4,1] --gridDim=[64,64]

// IMPERIAL EDIT: this kernel was commented out
#include "common.h"

__global__ void renderPixel(uint * result, Node * dnode, uint imageW, uint imageH, float pas, float df)
{
    __requires(imageW == 64);
    __requires(imageH == 64);
	uint id(blockIdx.x + __umul24(blockIdx.y, imageW));
	uint tid(threadIdx.x), x(blockIdx.x), y(blockIdx.y);
	Node node;
	float t(0.0f), tPixel;
	float4 Color(make_float4(0.0f,0.0f,0.0f,1.0f));
	matrice3x4 M(MView);
	Rayon R;
	Sphere s;
	__shared__ float T[numObj];
	__shared__ uint Obj;

	T[tid] = 10000.0f;
	
	if( x < imageW && y < imageH && tid < numObj ) {
		node = dnode[tid];
		if( tid == 0 ) result[id] = 0;
		tPixel = 2.0f/float(min(imageW,imageH));
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
		
		s = node.s;
		s.C.x += pas;

		if( node.fg )
			t = intersectionPlan(R,s.C,s.C);
		else
			t = intersectionSphere(R,s.C,s.r);

		T[tid] = t;

		__syncthreads();

		if( tid == 0 ) {
			float tmp(t);
			Obj = 0;
			for( int i(1); i < numObj; i++ ) {
				if( T[i] > 0.0f && ( tmp == 0.0f || T[i] < tmp ) ) {
					tmp = T[i];
					Obj = i;
				}
			}
		}

		__syncthreads();

		if( tid == Obj && t > 0.0f ) {
			s = node.s;
			s.C.x += pas;
			float3 P(R.A+R.u*t), L(normalize(make_float3(0,1,2)-P)), V(-1*R.u);
			float3 N(node.fg?getNormaleP(P):getNormale(P,s.C));
			if( dot(N,L) > 0.0f ) {
				Color = 0.5f*make_float4(s.R,s.V,s.B,s.A)*(max(0.0f,dot(N,L)));
            #ifdef FIXED_CONST_PARSE
				Color += 0.8f*make_float4(1.0f,1.0f,1.0f,1.0f)*pow(max(0.0f,min(1.0f,dot(2.0f*N*dot(N,L)-L,V))),20.0f);
            #else
            Color += 0.8f*make_float4(1.0f,1.0f,1.0f,1.0f)*float2int_pow20(max(0.0f,min(1.0f,dot(2.0f*N*dot(N,L)-L,V))));
            #endif
			}
			result[id] = rgbaFloatToInt(Color);
		}
	}

}
