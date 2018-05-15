//pass
//--local_size=[512] --num_groups=[782]

#include "../single_common.h"

static __attribute__((always_inline))
void cdfCalc(__global float * CDF, __global float * weights, int Nparticles)
{
	int x;
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}
}

__kernel void normalize_weights_kernel(__global float * weights, int Nparticles, __global float * partial_sums, __global float * CDF, __global float * u, __global int * seed)
{
  __requires(Nparticles == 400000);
	int i = get_global_id(0);
	int local_id = get_local_id(0);
	__local float u1;
	__local float sumWeights;

	if(0 == local_id)
		sumWeights = partial_sums[0];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(i < Nparticles) {
		weights[i] = weights[i]/sumWeights;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(i == 0) {
		cdfCalc(CDF, weights, Nparticles);
		u[0] = (1/((float)(Nparticles))) * d_randu(seed, i); // do this to allow all threads in all blocks to use the same u1
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(0 == local_id)
		u1 = u[0];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(i < Nparticles)
	{
		u[i] = u1 + i/((float)(Nparticles));
	}
}
