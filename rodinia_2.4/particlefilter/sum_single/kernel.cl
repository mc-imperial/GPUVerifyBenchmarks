//pass
//--global_size=400384 --local_size=512

__kernel void sum_kernel(__global float* partial_sums, int Nparticles)
{
  __requires(Nparticles == 400000);

	int i = get_global_id(0);
        size_t THREADS_PER_BLOCK = get_local_size(0);

	if(i == 0)
	{
		int x;
		float sum = 0;
#if 0 // imperial edit: rewrite around ceil
		int num_blocks = ceil((float) Nparticles / (float) THREADS_PER_BLOCK);
#else
		int num_blocks = (Nparticles / THREADS_PER_BLOCK) + 1;
#endif
		for (x = 0; x < num_blocks; x++) {
			sum += partial_sums[x];
		}
		partial_sums[0] = sum;
	}
}
