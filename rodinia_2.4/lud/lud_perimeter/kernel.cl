//pass
//--global_size=[2016,1] --local_size=[32,1]

// This is also used with num_groups=[62,1] .. [1,1]

#define BLOCK_SIZE 16

__kernel void
lud_perimeter(__global float *m, 
			  __local  float *dia,
			  __local  float *peri_row,
			  __local  float *peri_col,
			  int matrix_dim, 
			  int offset)
{
    __requires(matrix_dim == 1024);
    __requires(offset == 0);

    int i,j, array_offset;
    int idx;

    int  bx = get_group_id(0);	
    int  tx = get_local_id(0);

    if (tx < BLOCK_SIZE) {
      idx = tx;
      array_offset = offset*matrix_dim+offset;
      for (i=0;
           __global_invariant(__read_implies(m,
                                      (__read_offset_bytes(m)/sizeof(float) - idx) / matrix_dim < BLOCK_SIZE/2)),
           __global_invariant(__write_implies(dia,
                                      (__write_offset_bytes(dia)/sizeof(float) - tx) / BLOCK_SIZE < BLOCK_SIZE/2)),
           __invariant(array_offset == matrix_dim * i),
           i < BLOCK_SIZE/2; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
      }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0;
         __global_invariant(__read_implies(m,
                                      ((__read_offset_bytes(m)/sizeof(float) - (tx + (bx+1)*BLOCK_SIZE)) % matrix_dim == 0) |
                                      ((__read_offset_bytes(m)/sizeof(float) - tx) % matrix_dim == 0))),
         __global_invariant(__read_implies(m,
                                      ((__read_offset_bytes(m)/sizeof(float) - (tx + (bx+1)*BLOCK_SIZE)) / matrix_dim < BLOCK_SIZE/2) |
                                      ((__read_offset_bytes(m)/sizeof(float) - tx) / matrix_dim < BLOCK_SIZE))),
         __invariant(array_offset == matrix_dim * i),
         i < BLOCK_SIZE; i++) {
      peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

    } else {
    idx = tx-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2;
         __global_invariant(__implies(__read(m) & (tx >= BLOCK_SIZE),
                                      (__read_offset_bytes(m)/sizeof(float) - ((tx-BLOCK_SIZE) + (BLOCK_SIZE/2)*matrix_dim)) % matrix_dim == 0)),
         __global_invariant(__implies(__read(m) & (tx >= BLOCK_SIZE),
                                      (__read_offset_bytes(m)/sizeof(float) - ((tx-BLOCK_SIZE) + (BLOCK_SIZE/2)*matrix_dim)) / matrix_dim < BLOCK_SIZE/2)),
         __global_invariant(__implies(__write(dia) & (tx >= BLOCK_SIZE),
                                      (__write_offset_bytes(dia)/sizeof(float) - (tx-BLOCK_SIZE)) % BLOCK_SIZE == 0)),
         __invariant(array_offset == matrix_dim * (i - BLOCK_SIZE/2) + (BLOCK_SIZE/2)*matrix_dim),
         i < BLOCK_SIZE; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0;
         __global_invariant(__implies(__read(m) & (tx >= BLOCK_SIZE),
                                      ((__read_offset_bytes(m)/sizeof(float) - ((tx-BLOCK_SIZE) + (BLOCK_SIZE/2)*matrix_dim)) % matrix_dim == 0) |
                                      ((__read_offset_bytes(m)/sizeof(float) - ((tx-BLOCK_SIZE) + (bx+1)*BLOCK_SIZE*matrix_dim)) % matrix_dim == 0))),
         __global_invariant(__implies(__read(m) & (tx >= BLOCK_SIZE),
                                      ((__read_offset_bytes(m)/sizeof(float) - ((tx-BLOCK_SIZE) + (BLOCK_SIZE/2)*matrix_dim)) / matrix_dim < BLOCK_SIZE/2) |
                                      ((__read_offset_bytes(m)/sizeof(float) - ((tx-BLOCK_SIZE) + (bx+1)*BLOCK_SIZE*matrix_dim)) / matrix_dim < BLOCK_SIZE))),
         __global_invariant(__write_implies(peri_col,
                                      (__write_offset_bytes(peri_col)/sizeof(float) - (tx-BLOCK_SIZE)) % BLOCK_SIZE == 0)),
         __invariant(array_offset == matrix_dim * i + (bx+1)*BLOCK_SIZE*matrix_dim),
         i < BLOCK_SIZE; i++) {
      peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  
   }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tx < BLOCK_SIZE) { //peri-row
     idx=tx;
      for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
    }
    } else { //peri-col
     idx=tx - BLOCK_SIZE;
     for(i=0;
         __global_invariant(__write_implies(peri_col, (__write_offset_bytes(peri_col)/sizeof(float) - (idx * BLOCK_SIZE)) < BLOCK_SIZE)),
         __global_invariant(__read_implies(peri_col, (__read_offset_bytes(peri_col)/sizeof(float) - (idx * BLOCK_SIZE)) < BLOCK_SIZE)),
         i < BLOCK_SIZE; i++){
      for(j=0;
          __global_invariant(__write_implies(peri_col, (__write_offset_bytes(peri_col)/sizeof(float) - (idx * BLOCK_SIZE)) < BLOCK_SIZE)),
          __global_invariant(__read_implies(peri_col, (__read_offset_bytes(peri_col)/sizeof(float) - (idx * BLOCK_SIZE)) < BLOCK_SIZE)),
          j < i; j++)
        peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
      peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
     }
   }

	barrier(CLK_LOCAL_MEM_FENCE);
    
  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0;
        __global_invariant(__implies(__write(m) & (tx >= BLOCK_SIZE),
                                     (__write_offset_bytes(m)/sizeof(float) - ((tx - BLOCK_SIZE) + (bx+1)*BLOCK_SIZE*matrix_dim)) % matrix_dim == 0)),
        __global_invariant(__implies(__write(m) & (tx >= BLOCK_SIZE),
                                     (__write_offset_bytes(m)/sizeof(float) - ((tx - BLOCK_SIZE) + (bx+1)*BLOCK_SIZE*matrix_dim)) / matrix_dim < BLOCK_SIZE)),
        __invariant(array_offset == matrix_dim * i + (bx+1)*BLOCK_SIZE*matrix_dim),
        i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  }

}
