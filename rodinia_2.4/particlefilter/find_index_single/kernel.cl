//pass
//--local_size=[512] --num_groups=[782]

__kernel void find_index_kernel(__global float * arrayX, __global float * arrayY, 
	__global float * CDF, __global float * u, __global float * xj, 
	__global float * yj, __global float * weights, int Nparticles
	){
		int i = get_global_id(0);

		if(i < Nparticles){

			int index = -1;
			int x;

			for(x = 0; x < Nparticles; x++){
				if(CDF[x] >= u[i]){
					index = x;
					break;
				}
			}
			if(index == -1){
				index = Nparticles-1;
			}

			xj[i] = arrayX[index];
			yj[i] = arrayY[index];

			//weights[i] = 1 / ((float) (Nparticles)); //moved this code to the beginning of likelihood kernel

		}
		barrier(CLK_GLOBAL_MEM_FENCE);
}
