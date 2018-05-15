//pass
//--local_size=[1024] --num_groups=[2594]

#define GRIDSIZE_VAL1 256
#define SIZE_XY_VAL 65536

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

__kernel void binning_kernel (unsigned int n, 
                              __global ReconstructionSample* sample_g, 
                              __global unsigned int* idxKey_g,
                              __global unsigned int* idxValue_g, 
                              __global unsigned int* binCount_g, 
                              unsigned int binsize, unsigned int gridNumElems){
  unsigned int key;
  unsigned int sampleIdx = get_global_id(0); //blockIdx.x*blockDim.x+threadIdx.x;
  ReconstructionSample pt;
  unsigned int binIdx;
  unsigned int count;

  if (sampleIdx < n){
    pt = sample_g[sampleIdx];
    
    binIdx = (unsigned int)(pt.kZ)*((int) ( SIZE_XY_VAL )) + (unsigned int)(pt.kY)*((int)( GRIDSIZE_VAL1 )) + (unsigned int)(pt.kX);

    count = atom_add(binCount_g+binIdx, 1);
    if (count < binsize){
      key = binIdx;
    } else {
      atom_sub(binCount_g+binIdx, 1);
      key = gridNumElems;
    }

    idxKey_g[sampleIdx] = key;
    idxValue_g[sampleIdx] = sampleIdx;
  }
}
