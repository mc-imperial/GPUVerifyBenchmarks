//pass
//--global_size=494080 --local_size=256

// The bug both causes a race and an out-of-bounds error.
// To expose the out-of-bounds error also pass the following
// to GPUVerify:
//--kernel-arrays=kmeans_swap,67186720,67186720 --check-array-bounds

__kernel void
kmeans_swap(__global float  *feature,   
			__global float  *feature_swap,
			int     npoints,
			int     nfeatures
){
  __requires(npoints == 494020);
  __requires(nfeatures == 34);

	unsigned int tid = get_global_id(0);
#ifndef KERNEL_BUG
  if (tid < npoints)
#endif
	for(int i = 0;
        __global_invariant(__write_implies(feature_swap, __write_offset_bytes(feature_swap)/sizeof(float)%npoints == tid)),
        i <  nfeatures; i++)
		feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];

} 
