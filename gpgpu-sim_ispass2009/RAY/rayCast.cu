//pass
//--blockDim=32 --gridDim=32

// IMPERIAL EDIT: this kernel was commented out
#include "common.h"

__global__ void rayCast (uint * d_output, uint * d_temp, uint imageW, uint imageH, float pas, float df)
//(uint * result, uint * temp, uint imageW, uint imageH, float pas, float df)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y);
	uint id = x + y * gridDim.x;
	//float tmp= float(imageW)/float(gridDim.x);
	float t;

	//if( x < gridDim.x && y < gridDim.y )
	if( d_temp[id] == 0 )
	{
		float tPixel = 2.0f/float(imageW);
		matrice3x4 M(MView);
		Rayon R;
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
		Sphere s(cnode[1].s);
		s.C.x += pas;
		t = intersectionSphere(R,s.C,s.r/(imageW/gridDim.x));

		if( t > 0.0f ) {		
			//float4 f = make_float4(0,1,0,1)*(dot(getNormale(R.A+R.u*t,s.C),(-1.0f)*R.u));
			d_output[id] = rgbaFloatToInt(make_float4(0,1,0,1));
			//printf("%d %d\n",int(x*tmp),int((y*tmp)/2));
		}
		else {
//       float tmp= float(imageW)/gridDim.x;
//       d_temp[int(x*tmp+(y*tmp)*imageW)] = 1;
//       d_temp[int(x*tmp+(tmp*(float(y)+0.5f)*imageW))] = 1;
//       d_temp[int(tmp*(float(x)+0.5f)+(y*tmp)*imageW)] = 1;
//       d_temp[int(tmp*(float(x)+0.5f)+(tmp*(float(y)+0.5f)*imageW))] = 1;
			//if(gridDim.x==16) printf("hep %d %f\n",gridDim.x,t);
		}
	}
	else {
//       float tmp= float(imageW)/gridDim.x;
//       d_temp[int(x*tmp+(y*tmp)*imageW)] = 1;
//       d_temp[int(x*tmp+(tmp*(float(y)+0.5f)*imageW))] = 1;
//       d_temp[int(tmp*(float(x)+0.5f)+(y*tmp)*imageW)] = 1;
//       d_temp[int(tmp*(float(x)+0.5f)+(tmp*(float(y)+0.5f)*imageW))] = 1;
			//if(gridDim.x==16) printf("hep %d %f\n",gridDim.x,t);
	}
	//__syncthreads();
}
