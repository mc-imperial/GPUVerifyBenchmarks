//pass
//--local_size=[32,1] --num_groups=[11,9]

#include "../common.h"

__kernel void larger_sad_calc_16(__global unsigned short *blk_sad,
				   int mb_width,
				   int mb_height)
{
  __requires(mb_width == 11 /* get_num_groups(0) */);
  __requires(mb_height == 9 /* get_num_groups(1) */);
  // Macroblock coordinates 
  int mb_x = get_group_id(0);
  int mb_y = get_group_id(1);

  // Number of macroblocks in a frame
  int macroblocks = mul24(mb_width, mb_height) * MAX_POS_PADDED;
  int macroblock_index = (mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

  __global unsigned short *bi;
  __global unsigned short *bo_3, *bo_2, *bo_1;

  //bi = blk_sad + macroblocks * 5 + macroblock_index * 4;
  bi = blk_sad + ((macroblocks + macroblock_index) << 2) + macroblocks;

  // Block type 3: 8x16
  //bo_3 = blk_sad + macroblocks * 3 + macroblock_index * 2;
  bo_3 = blk_sad + ((macroblocks + macroblock_index) << 1) + macroblocks;

  // Block type 5: 8x4
  bo_2 = blk_sad + macroblocks + macroblock_index * 2;

  // Block type 4: 8x8
  bo_1 = blk_sad + macroblock_index;

  for (int search_pos = get_local_id(0);
       __global_invariant(__read_implies(blk_sad, ((__read_offset_bytes(blk_sad) - __ptr_offset_bytes(bi))/sizeof(ushort2) < (MAX_POS+1)/2)
                                                | ((__read_offset_bytes(blk_sad) - __ptr_offset_bytes(bi))/sizeof(ushort2) - (MAX_POS_PADDED/2) < (MAX_POS+1)/2)
                                                | ((__read_offset_bytes(blk_sad) - __ptr_offset_bytes(bi))/sizeof(ushort2) - (2*MAX_POS_PADDED/2) < (MAX_POS+1)/2)
                                                | ((__read_offset_bytes(blk_sad) - __ptr_offset_bytes(bi))/sizeof(ushort2) - (3*MAX_POS_PADDED/2) < (MAX_POS+1)/2))),
       __global_invariant(__write_implies(blk_sad, (((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_3))/sizeof(ushort2) < (MAX_POS+1)/2)
                                                  & ((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_3))/sizeof(ushort2)%32 == get_local_id(0)))
                                                |  (((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_3))/sizeof(ushort2) - (MAX_POS_PADDED/2) < (MAX_POS+1)/2)
                                                  & (((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_3))/sizeof(ushort2) - (MAX_POS_PADDED/2))%32 == get_local_id(0)))
                                                |  (((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_2))/sizeof(ushort2) < (MAX_POS+1)/2)
                                                  & ((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_2))/sizeof(ushort2)%32 == get_local_id(0)))
                                                |  (((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_2))/sizeof(ushort2) - (MAX_POS_PADDED/2) < (MAX_POS+1)/2)
                                                  & (((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_2))/sizeof(ushort2) - (MAX_POS_PADDED/2))%32 == get_local_id(0)))
                                                |  (((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_1))/sizeof(ushort2) < (MAX_POS+1)/2)
                                                  & ((__write_offset_bytes(blk_sad) - __ptr_offset_bytes(bo_1))/sizeof(ushort2)%32 == get_local_id(0))))),
       search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
#if SHORT2_V
  #if VEC_LOAD
      ushort2 s00 = vload2(search_pos,                    bi);
      ushort2 s01 = vload2(search_pos+  MAX_POS_PADDED/2, bi);
      ushort2 s10 = vload2(search_pos+2*MAX_POS_PADDED/2, bi);
      ushort2 s11 = vload2(search_pos+3*MAX_POS_PADDED/2, bi);
  #else
      ushort2 s00 = (ushort2) (bi[search_pos*2], bi[search_pos*2+1]);
      ushort2 s01 = (ushort2) (bi[(search_pos + MAX_POS_PADDED/2)*2], bi[(search_pos + MAX_POS_PADDED/2)*2+1]);
      ushort2 s10 = (ushort2) (bi[(search_pos + 2*MAX_POS_PADDED/2)*2], bi[(search_pos + 2*MAX_POS_PADDED/2)*2+1]);
      ushort2 s11 = (ushort2) (bi[(search_pos + 3*MAX_POS_PADDED/2)*2], bi[(search_pos + 3*MAX_POS_PADDED/2)*2+1]);
  #endif

  #if VEC_STORE
      ushort2 s0010 = s00 + s10;
      ushort2 s0111 = s01 + s11;
      ushort2 s0001 = s00 + s01;
      ushort2 s1011 = s10 + s11;
      ushort2 s00011011 = s0001 + s1011;
      
      vstore2(s0010, search_pos, bo_3);
      vstore2(s0111, search_pos+MAX_POS_PADDED/2, bo_3);
      vstore2(s0001, search_pos, bo_2);
      vstore2(s1011, search_pos+MAX_POS_PADDED/2, bo_2);
      vstore2(s00011011, search_pos, bo_1);
  #elif CAST_STORE
      ((__global ushort2 *)bo_3)[search_pos]                  = s00 + s10;
      ((__global ushort2 *)bo_3)[search_pos+MAX_POS_PADDED/2] = s01 + s11;
      ((__global ushort2 *)bo_2)[search_pos]                  = s00 + s01;
      ((__global ushort2 *)bo_2)[search_pos+MAX_POS_PADDED/2] = s10 + s11;
      ((__global ushort2 *)bo_1)[search_pos]                  = (s00 + s01) + (s10 + s11);
  #else // SCALAR_STORE
      bo_3[search_pos*2] = s00.x + s10.x;
      bo_3[search_pos*2+1] = s00.y + s10.y;
      bo_3[(search_pos+MAX_POS_PADDED/2)*2] = s01.x + s11.x;
      bo_3[(search_pos+MAX_POS_PADDED/2)*2+1] = s01.y + s11.y;
      bo_2[search_pos*2] = s00.x + s01.x;
      bo_2[search_pos*2+1] = s00.y + s01.y;
      bo_2[(search_pos+MAX_POS_PADDED/2)*2] = s10.x + s11.x;
      bo_2[(search_pos+MAX_POS_PADDED/2)*2+1] = s10.y + s11.y;
      bo_1[search_pos*2] = (s00.x + s01.x) + (s10.x + s11.x);
      bo_1[search_pos*2+1] = (s00.y + s01.y) + (s10.y + s11.y);
  #endif
#else // UINT_CUDA_V
      uint i00 = ((__global uint *)bi)[search_pos];
      uint i01 = ((__global uint *)bi)[search_pos + MAX_POS_PADDED/2];
      uint i10 = ((__global uint *)bi)[search_pos + 2*MAX_POS_PADDED/2];
      uint i11 = ((__global uint *)bi)[search_pos + 3*MAX_POS_PADDED/2];

      ((__global uint *)bo_3)[search_pos]                  = i00 + i10;
      ((__global uint *)bo_3)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
      ((__global uint *)bo_2)[search_pos]                  = i00 + i01;
      ((__global uint *)bo_2)[search_pos+MAX_POS_PADDED/2] = i10 + i11;
      ((__global uint *)bo_1)[search_pos]                  = (i00 + i01) + (i10 + i11);
#endif
    }
}
