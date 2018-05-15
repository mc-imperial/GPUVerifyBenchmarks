//pass
//--num_groups=128 --local_size=64

#include "../common.h"

__kernel void fft1D_512 (__global T2 *work)
{
  int tid = get_local_id(0); 
  int blockIdx = get_group_id(0) * 512 + tid; 
  int hi = tid>>3;
  int lo = tid&7;
  T2 data[8]; 
  __local T smem[8*8*9];

  // starting index of data to/from global memory 
  work = work + blockIdx;  
  //out = out + blockIdx; 
  globalLoads8(data, work, 64); // coalesced global reads 

  FFT8( data );

  twiddle8( data, tid, 512 );
  transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);

  FFT8( data );

  twiddle8( data, hi, 64 );
  transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);

  FFT8( data );

  globalStores8(data, work, 64);
}
