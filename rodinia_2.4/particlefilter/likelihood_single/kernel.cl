//pass
//--local_size=[512] --num_groups=[782]

#include "../single_common.h"

#define SCALE_FACTOR 300

static __attribute__((always_inline))
float d_randn(__global int * seed, int index){
	//Box-Muller algortihm
	float pi = 3.14159265358979323846;
	float u = d_randu(seed, index);
	float v = d_randu(seed, index);
	float cosine = cos(2*pi*v);
	float rt = -2*log(u);
	return sqrt(rt)*cosine;
}

static __attribute__((always_inline))
float calcLikelihoodSum(__global unsigned char * I, __global int * ind, int numOnes, int index){
	float likelihoodSum = 0.0;
	int x;
	for(x = 0; x < numOnes; x++)
		likelihoodSum += (pow((float)(I[ind[index*numOnes + x]] - 100),2) - pow((float)(I[ind[index*numOnes + x]]-228),2))/50.0;
	return likelihoodSum;
}

static __attribute__((always_inline))
float dev_round_float(float value) {
    int newValue = (int) (value);
    if (value - newValue < .5f)
        return newValue;
    else
        return newValue++;
}

__kernel void likelihood_kernel(__global float * arrayX, __global float * arrayY,__global float * xj, __global float * yj, __global float * CDF, __global int * ind, __global int * objxy, __global float * likelihood, __global unsigned char * I, __global float * u, __global float * weights, const int Nparticles, const int countOnes, const int max_size, int k, const int IszY, const int Nfr, __global int *seed, __global float * partial_sums, __local float* buffer){
  __requires(Nparticles == 400000);
  __requires(countOnes == 69);
	int block_id = get_group_id(0);
	int thread_id = get_local_id(0);
	int i = get_global_id(0);
        size_t THREADS_PER_BLOCK = get_local_size(0);
	int y;
	int indX, indY;
	
	
	if(i < Nparticles){
		arrayX[i] = xj[i]; 
		arrayY[i] = yj[i]; 

		weights[i] = 1 / ((float) (Nparticles)); //Donnie - moved this line from end of find_index_kernel to prevent all weights from being reset before calculating position on final iteration.


		arrayX[i] = arrayX[i] + 1.0 + 5.0*d_randn(seed, i);
		arrayY[i] = arrayY[i] - 2.0 + 2.0*d_randn(seed, i);

	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(i < Nparticles)
	{
		for(y = 0;
            __global_invariant(__write_implies(ind, __write_offset_bytes(ind)/sizeof(int) - i * countOnes < countOnes)),
            y < countOnes; y++){

			indX = dev_round_float(arrayX[i]) + objxy[y*2 + 1];
			indY = dev_round_float(arrayY[i]) + objxy[y*2];

			ind[i*countOnes + y] = abs(indX*IszY*Nfr + indY*Nfr + k);
			if(ind[i*countOnes + y] >= max_size)
				ind[i*countOnes + y] = 0;
		}
		likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);

		likelihood[i] = likelihood[i]/countOnes-SCALE_FACTOR;

		weights[i] = weights[i] * exp(likelihood[i]); //Donnie Newell - added the missing exponential function call

	}
	
	buffer[thread_id] = 0.0; // DEBUG!!!!!!!!!!!!!!!!!!!!!!!!
	//buffer[thread_id] = i;
		
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	if(i < Nparticles){
		buffer[thread_id] = weights[i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	/* for some reason the get_local_size(0) call was not returning 512. */
	//for(unsigned int s=get_local_size(0)/2; s>0; s>>=1)
	for(unsigned int s=THREADS_PER_BLOCK/2; s>0; s>>=1)
	{
		if(thread_id < s)
		{
			buffer[thread_id] += buffer[thread_id + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(thread_id == 0)
	{
		partial_sums[block_id] = buffer[0];
	}
	
}//*/
