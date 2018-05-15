//pass
//--blockDim=32 --gridDim=32

// IMPERIAL EDIT: this kernel was commented out
#include "common.h"

__global__ void rayTrace(uint * Obj, float * prof, float3 * A, float3 * u, uint imageW, uint imageH, float pas, float df, uint nObj)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint id = x + y * imageW;

	if( x < imageW && y < imageH )
	{
		Sphere s(cnode[nObj].s);
		float t;
		s.C.x += pas;
		Rayon R;
		R.A = A[id];
		R.u = u[id];
		t = intersectionSphere(R,s.C,s.r);

		if( t > 0.0f && t < prof[id] ) {
			prof[id] = t;
			Obj[id] = nObj;
		}
	}
}
