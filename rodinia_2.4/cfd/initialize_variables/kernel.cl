//pass
//--global_size=[97152,1] --local_size=[192,1]

#include "../common.h"

__kernel void initialize_variables(__global float* variables, __constant float* ff_variable, int nelr){
  __requires(nelr == 97152);
	//const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	const int i = get_global_id(0);
	for(int j = 0;
        __invariant(__write_implies(variables, __write_offset_bytes(variables)/sizeof(float)%nelr == get_global_id(0))),
        j < NVAR; j++)
		variables[i + j*nelr] = ff_variable[j];
	
}
