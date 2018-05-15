//pass
//--blockDim=32 --gridDim=32

// IMPERIAL EDIT: this kernel was commented out
#include "common.h"

/*__global__  __device__ void rayCalc(float3 * A, float3 * u, float * prof, uint imageW, uint imageH, float df, float tPixel)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint id = x + y * imageW;

	if( x < imageW && y < imageH )
	{
		matrice3x4 M(MView);
		Rayon R;
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
		A[id] = R.A;
		u[id] = R.u;
		prof[id] = 1000.0f;
	}
}*/
