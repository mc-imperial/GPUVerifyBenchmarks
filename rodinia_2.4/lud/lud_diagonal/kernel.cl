//pass
//--global_size=[16,1] --local_size=[16,1]

#define BLOCK_SIZE 16

__kernel void 
lud_diagonal(__global float *m, 
			 __local  float *shadow,
			 int   matrix_dim, 
			 int   offset)
{ 
  __requires(matrix_dim == 1024);
  __requires(offset == 0);

	int i,j;
	int tx = get_local_id(0);

	int array_offset = offset*matrix_dim+offset;
	for(i=0; i < BLOCK_SIZE; i++){
		shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
		array_offset += matrix_dim;
	}
  
	barrier(CLK_LOCAL_MEM_FENCE);
  
	for(i=0; i < BLOCK_SIZE-1; i++) {

    if (tx>i){
      for(j=0;
          __global_invariant(__read_implies(shadow,
                                              ((__read_offset_bytes(shadow)/sizeof(float)) - (tx * BLOCK_SIZE ) <= i)
                                            | ((__read_offset_bytes(shadow)/sizeof(float) - i) / BLOCK_SIZE < i))),
          j < i; j++)
        shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
		shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }

	barrier(CLK_LOCAL_MEM_FENCE);
    if (tx>i){

      for(j=0;
          __global_invariant(__read_implies(shadow,
                                              ((__read_offset_bytes(shadow)/sizeof(float) - tx) % BLOCK_SIZE == 0)
                                            | ((__read_offset_bytes(shadow)/sizeof(float)) - (BLOCK_SIZE * (i + 1)) < i + 1))),
          j < i+1; j++)
        shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
    }
    
	barrier(CLK_LOCAL_MEM_FENCE);
    }

    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
      array_offset += matrix_dim;
    }
  
}
