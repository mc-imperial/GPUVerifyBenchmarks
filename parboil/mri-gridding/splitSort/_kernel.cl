//pass
//--local_size=[256] --num_groups=[2594]

#define UINT32_MAX 0xffffffff

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable 

#define BITS 4
#define LNB 4

#define SORT_BS 256

#define CONFLICT_FREE_OFFSET(index) (((unsigned int)(index) >> min((unsigned int)((LNB)+(index)), (unsigned int)(32-(2*LNB))))>>(2*LNB)) // In the original, the type conversion of the first argument to min is poorly scoped, and so doesn't typecheck with AMDAPP 2.8.1. Fixed here.
#define BLOCK_P_OFFSET (4*SORT_BS+1+(4*SORT_BS+1)/16+(4*SORT_BS+1)/64)

static __attribute__((always_inline))
void scan (__local unsigned int* s_data){
  unsigned int thid = get_local_id(0);

  barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

  s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] += s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)];
  s_data[2*(get_local_size(0)+thid)+1+CONFLICT_FREE_OFFSET(2*(get_local_size(0)+thid)+1)] += s_data[2*(get_local_size(0)+thid)+CONFLICT_FREE_OFFSET(2*(get_local_size(0)+thid))];

  unsigned int stride = 2;
  for (unsigned int d = get_local_size(0); d > 0; d >>= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    if (thid < d)
    {
      unsigned int i  = 2*stride*thid;
      unsigned int ai = i + stride - 1;
      unsigned int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      s_data[bi] += s_data[ai];
    }

    stride *= 2;
  }

  if (thid == 0){
    unsigned int last = 4*get_local_size(0)-1;
    last += CONFLICT_FREE_OFFSET(last);
    s_data[4*get_local_size(0)+CONFLICT_FREE_OFFSET(4*get_local_size(0))] = s_data[last];
    s_data[last] = 0;
  }

  for (unsigned int d = 1; d <= get_local_size(0); d *= 2)
  {
    stride >>= 1;

    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    if (thid < d)
    {
      unsigned int i  = 2*stride*thid;
      unsigned int ai = i + stride - 1;
      unsigned int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      unsigned int t  = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

  unsigned int temp = s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)];
  s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)] = s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)];
  s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] += temp;

  unsigned int temp2 = s_data[2*(get_local_size(0)+thid)+CONFLICT_FREE_OFFSET(2*(get_local_size(0)+thid))];
  s_data[2*(get_local_size(0)+thid)+CONFLICT_FREE_OFFSET(2*(get_local_size(0)+thid))] = s_data[2*(get_local_size(0)+thid)+1+CONFLICT_FREE_OFFSET(2*(get_local_size(0)+thid)+1)];
  s_data[2*(get_local_size(0)+thid)+1+CONFLICT_FREE_OFFSET(2*(get_local_size(0)+thid)+1)] += temp2;

  barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();
}

__kernel void splitSort(int numElems, int iter, 
                                 __global unsigned int* keys, 
                                 __global unsigned int* values, 
                                 __global unsigned int* histo)
{
    //IMPERIAL EDIT: a precondition is necessary; perhaps __requires(iter == 16);
    
    __local unsigned int flags[BLOCK_P_OFFSET];
    __local unsigned int histo_s[1<<BITS];

    const unsigned int tid = get_local_id(0);
    const unsigned int gid = get_group_id(0)*4*SORT_BS+4*get_local_id(0);

    // Copy input to shared mem. Assumes input is always even numbered
    uint4 lkey = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint4 lvalue;
    if (gid < numElems){
      lkey = *((__global uint4*)(keys+gid));
      lvalue = *((__global uint4*)(values+gid));
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    atom_add(histo_s+((lkey.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);

    uint4 index = (uint4) (4*tid, 4*tid+1, 4*tid+2, 4*tid+3);

    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint4 flag = (uint4) ( (lkey.x>>i)&0x1,(lkey.y>>i)&0x1,(lkey.z>>i)&0x1,(lkey.w>>i)&0x1 );

      flags[index.x+CONFLICT_FREE_OFFSET(index.x)] = 1<<(16*flag.x);
      flags[index.y+CONFLICT_FREE_OFFSET(index.y)] = 1<<(16*flag.y);
      flags[index.z+CONFLICT_FREE_OFFSET(index.z)] = 1<<(16*flag.z);
      flags[index.w+CONFLICT_FREE_OFFSET(index.w)] = 1<<(16*flag.w);

      scan (flags);

      index.x = (flags[index.x+CONFLICT_FREE_OFFSET(index.x)]>>(16*flag.x))&0xFFFF;
      index.y = (flags[index.y+CONFLICT_FREE_OFFSET(index.y)]>>(16*flag.y))&0xFFFF;
      index.z = (flags[index.z+CONFLICT_FREE_OFFSET(index.z)]>>(16*flag.z))&0xFFFF;
      index.w = (flags[index.w+CONFLICT_FREE_OFFSET(index.w)]>>(16*flag.w))&0xFFFF;

      unsigned short offset = flags[4*get_local_size(0)+CONFLICT_FREE_OFFSET(4*get_local_size(0))]&0xFFFF;
      index.x += (flag.x) ? offset : 0;
      index.y += (flag.y) ? offset : 0;
      index.z += (flag.z) ? offset : 0;
      index.w += (flag.w) ? offset : 0;

      barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();
    }

    // Write result.
    if (gid < numElems){
      keys[get_group_id(0)*4*SORT_BS+index.x] = lkey.x;
      keys[get_group_id(0)*4*SORT_BS+index.y] = lkey.y;
      keys[get_group_id(0)*4*SORT_BS+index.z] = lkey.z;
      keys[get_group_id(0)*4*SORT_BS+index.w] = lkey.w;

      values[get_group_id(0)*4*SORT_BS+index.x] = lvalue.x;
      values[get_group_id(0)*4*SORT_BS+index.y] = lvalue.y;
      values[get_group_id(0)*4*SORT_BS+index.z] = lvalue.z;
      values[get_group_id(0)*4*SORT_BS+index.w] = lvalue.w;
    }
    if (tid < (1<<BITS)){
      histo[get_num_groups(0)*get_local_id(0)+get_group_id(0)] = histo_s[tid];
    }
}
