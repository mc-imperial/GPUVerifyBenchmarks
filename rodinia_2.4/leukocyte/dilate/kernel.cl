//pass
//--global_size=140272 --local_size=176

// Kernel to compute the dilation of the GICOV matrix produced by the GICOV kernel
// Each element (i, j) of the output matrix is set equal to the maximal value in
//  the neighborhood surrounding element (i, j) in the input matrix
// Here the neighborhood is defined by the structuring element (c_strel)
#ifdef USE_IMAGE
__kernel void dilate_kernel(int img_m, int img_n, int strel_m, int strel_n, __constant float *c_strel,
                            image2d_t img, __global float *dilated) {
#else
__kernel void dilate_kernel(int img_m, int img_n, int strel_m, int strel_n, __constant float *c_strel,
                            __global float *img, __global float *dilated) {
#endif

  __requires(img_m == 219);
  __requires(img_n == 640);
  __requires(strel_m == 25);
  __requires(strel_n == 25);
        
	// Find the center of the structuring element
	int el_center_i = strel_m / 2;
	int el_center_j = strel_n / 2;

	// Determine this thread's location in the matrix
	int thread_id = get_global_id(0); //(blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id % img_m;
	int j = thread_id / img_m;

	// Initialize the maximum GICOV score seen so far to zero
	float max = 0.0f;
	
	#ifdef USE_IMAGE
	// Define the sampler for accessing the image
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	#endif

#ifndef KERNEL_BUG
  if (j < img_n) {
#endif

	// Iterate across the structuring element in one dimension
	int el_i, el_j, x, y;
	for (el_i = 0; el_i < strel_m; el_i++) {
		y = i - el_center_i + el_i;
		// Make sure we have not gone off the edge of the matrix
		if ( (y >= 0) && (y < img_m) ) {
			// Iterate across the structuring element in the other dimension
			for (el_j = 0; el_j < strel_n; el_j++) {
				x = j - el_center_j + el_j;
				// Make sure we have not gone off the edge of the matrix
				//  and that the current structuring element value is not zero
				if ( (x >= 0) &&
					 (x < img_n) &&
					 (c_strel[(el_i * strel_n) + el_j] != 0) ) {
						// Determine if this is the maximal value seen so far
						#ifdef USE_IMAGE
						int2 addr = {y, x};
						float temp = read_imagef(img, sampler, addr).x;
						#else
						int addr = (x * img_m) + y;
						float temp = img[addr];
						#endif
						if (temp > max) max = temp;
				}
			}
		}
	}
	
	// Store the maximum value found
	dilated[(i * img_n) + j] = max;

#ifndef KERNEL_BUG
  }
#endif
}
