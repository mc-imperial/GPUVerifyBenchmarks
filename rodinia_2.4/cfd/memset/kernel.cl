//pass
//--local_size=[192,1] --num_groups=[2024,1]

#include "../common.h"

__kernel void memset_kernel(__global char * mem_d, short val, int number_bytes){
	const int thread_id = get_global_id(0);
	mem_d[thread_id] = val;
}
