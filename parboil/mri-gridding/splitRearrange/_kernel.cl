//pass
//--local_size=[256] --num_groups=[2594]

/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
 
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable 

#define UINT32_MAX 4294967295
#define BITS 4
#define LNB 4

#define SORT_BS 256

__kernel void splitRearrange (int numElems, int iter, 
                                __global unsigned int* keys_i, 
                                __global unsigned int* keys_o, 
                                __global unsigned int* values_i, 
                                __global unsigned int* values_o, 
                                __global unsigned int* histo){
  __local unsigned int histo_s[(1<<BITS)];
  __local uint array_s[4*SORT_BS];
  int index = get_group_id(0)*4*SORT_BS + 4*get_local_id(0);

  if (get_local_id(0) < (1<<BITS)){
    histo_s[get_local_id(0)] = histo[get_num_groups(0)*get_local_id(0)+get_group_id(0)];
  }

  uint4 mine, value;
  if (index < numElems){
    mine = *((__global uint4*)(keys_i+index));
    value = *((__global uint4*)(values_i+index));
  } else {
    mine.x = UINT32_MAX;
    mine.y = UINT32_MAX;
    mine.z = UINT32_MAX;
    mine.w = UINT32_MAX;
  }
  
  uint4 masks = (uint4) ( (mine.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter) );

//  ((__local uint4*)array_s)[get_local_id(0)] = masks;
  vstore4(masks, get_local_id(0), (__local uint *)array_s);
  
  barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

  uint4 new_index = (uint4) ( histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w] );

  int i = 4*get_local_id(0)-1;
  
  while (i >= 0){
    if (array_s[i] == masks.x){
      new_index.x++;
      i--;
    } else {
      break;
    }
  }

  new_index.y = (masks.y == masks.x) ? new_index.x+1 : new_index.y;
  new_index.z = (masks.z == masks.y) ? new_index.y+1 : new_index.z;
  new_index.w = (masks.w == masks.z) ? new_index.z+1 : new_index.w;

  if (index < numElems){
    keys_o[new_index.x] = mine.x;
    values_o[new_index.x] = value.x;

    keys_o[new_index.y] = mine.y;
    values_o[new_index.y] = value.y;

    keys_o[new_index.z] = mine.z;
    values_o[new_index.z] = value.z;

    keys_o[new_index.w] = mine.w;
    values_o[new_index.w] = value.w; 
  }  
}

