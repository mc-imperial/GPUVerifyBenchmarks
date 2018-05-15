//pass
//--local_size=[1024] --num_groups=[2594]

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable 
 
typedef struct{
  float real;
  float imag;
  float kX;
  float kY;
  float kZ;
  float sdc;
} ReconstructionSample;

#define TILE 64
#define LOG_TILE 6

__kernel void reorder_kernel(int n, 
                               __global unsigned int* idxValue_g, 
                               __global ReconstructionSample* samples_g, 
                               __global ReconstructionSample* sortedSample_g){
  unsigned int index = get_global_id(0); //blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int old_index;
  ReconstructionSample pt;

  if (index < n){
    old_index = idxValue_g[index];
    pt = samples_g[old_index];
    sortedSample_g[index] = pt;
  }
}

