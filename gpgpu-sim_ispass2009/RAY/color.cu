//pass
//--blockDim=32 --gridDim=32

// IMPERIAL EDIT: this kernel was commented out
#include "common.h"

__global__  void color(uint * result, uint * Obj, float * prof, float3 * A, float3 * u, uint imageW, uint imageH, float pas)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint id = x + y * imageW;

	if( x < imageW && y < imageH )
	{
		float t(prof[id]);
		if( t > 0.0f  && t < 1000.0f ) {
			Rayon R;
			R.A = A[id];
			R.u = u[id];
			Sphere s(cnode[Obj[id]].s);
			s.C.x += pas;
			float4 f = make_float4(s.R,s.V,s.B,s.A)*(dot(getNormale(R.A+R.u*t,s.C),(-1.0f)*R.u));
			result[id] = rgbaFloatToInt(f);
		}
		else {
			result[id] = 0;
		}
		prof[id] = 100000.0f;
	}
}
