//pass
//--num_groups=128 --local_size=64

#include "../common.h"

__kernel void ifft1D_512 (__global T2 *work)
{
  int i;
  int tid = get_local_id(0); 
  int blockIdx = get_group_id(0) * 512 + tid; 
  int hi = tid>>3;
  int lo = tid&7;
  T2 data[8]; 
  __local T smem[8*8*9];
  
  // starting index of data to/from global memory 
  work = work + blockIdx; 
  globalLoads8(data, work, 64); // coalesced global reads 

  // Inject an artificial error for testing the sensitivity of FFT
  // if( blockIdx == 0 ){ data[6] *= 1.001; }

  IFFT8( data );

  itwiddle8( data, tid, 512 );
  transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);

  IFFT8( data );

  itwiddle8( data, hi, 64 );
  transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);

  IFFT8( data );

  for(i=0; i<8; i++) {
      data[i].x = data[i].x/512.0f;
      data[i].y = data[i].y/512.0f;
  }

  globalStores8(data, work, 64);

}
