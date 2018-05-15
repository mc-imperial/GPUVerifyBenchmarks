//pass
//--local_size=[256,1] --num_groups=[12288,1]

// Also uses num_groups=[2048,1] and [256,1]

__kernel void memset_kernel(__global char * mem_d, short val, int number_bytes){
	const int thread_id = get_global_id(0);
	mem_d[thread_id] = val;
}
