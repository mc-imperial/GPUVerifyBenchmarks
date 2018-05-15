//pass
//--blockDim=32 --gridDim=32

// IMPERIAL EDIT: this kernel was commented out
#include "common.h"

__global__ void d_render(uint * d_output, uint imageW, uint imageH, float pas, float df, float tPixel)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint id = x + y * imageW;

	if( x < imageW && y < imageH )
	{
		//float tPixel = 2.0f/(float)min(imageW,imageH);
		matrice3x4 M(MView);
		Rayon R;
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
		Sphere s(cnode[1].s), s2(cnode[2].s), st(cnode[2].s);
		float t, t2, tt;
		s.C.x += pas, s2.C.x += pas;
		t = intersectionSphere(R,s.C,s.r);
		t2 = intersectionSphere(R,s2.C,s2.r);
		if( !t ) {
			//myswap(s,s2);
			//swap(t,t2);
         tt = t;
			t = t2;
			t2 = tt;
			st = s;
			s = s2;
			s2 = st;
		}
		else if( t2 && t2 < t ) {
			//myswap(s,s2);
			//swap(t,t2);
         tt = t;
			t = t2;
			t2 = tt;
         st = s;
         s = s2;
         s2 = st;
		}
		float4 f = make_float4(0,1,0,1)*(dot(getNormale(R.A+R.u*t,s.C),(-1.0f)*R.u));
		uint n = rgbaFloatToInt(f);
		//printf("%f\n",d_node[0].s.r);
		if( t > 0.0f )
			d_output[id] = n;
		//else d_output[id] = 0;
	}
	__syncthreads();
}
