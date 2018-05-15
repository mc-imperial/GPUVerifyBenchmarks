//pass
//--local_size=[61,1] --num_groups=[44,36]

#include "../common.h"

__kernel void mb_sad_calc(__global unsigned short *blk_sad,
                            __global unsigned short *frame,
                            int mb_width,
                            int mb_height,
                            __read_only image2d_t img_ref)
{
  __requires(mb_width == 11);
  __requires(mb_height == 9);
	const sampler_t texSampler =
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;


  int tx = (get_local_id(0) / CEIL_POS) % THREADS_W;
  int ty = (get_local_id(0) / CEIL_POS) / THREADS_W;
  int bx = get_group_id(0);
  int by = get_group_id(1);
  int img_width = mb_width*16;

  // Macroblock and sub-block coordinates
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  // If this thread is assigned to an invalid 4x4 block, do nothing 
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      // Pixel offset of the origin of the current 4x4 block
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      // Origin of the search area for this 4x4 block
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      // Origin in the current frame for this 4x4 block
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (get_local_id(0) % CEIL_POS) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      // All SADs from this thread are stored in a contiguous chunk
      // of memory starting at this offset
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      // Don't go past bounds
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      #define INV_OFFSET (mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +  \
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 + \
        (4 * block_y + block_x) * MAX_POS_PADDED)

      // For each search position, within the range allocated to this thread
      for (search_pos = search_pos_base;
           __global_invariant(__write_implies(blk_sad, __write_offset_bytes(blk_sad)/sizeof(unsigned short) - INV_OFFSET >= search_pos_base)),
           __global_invariant(__write_implies(blk_sad, __write_offset_bytes(blk_sad)/sizeof(unsigned short) - INV_OFFSET < search_pos_end)),
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        // 4x4 SAD computation
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
          
          // ([unsigned] short)read_imageui or
          //                   read_imagei  is required for correct calculation.
          // Though read_imagei() is shorter, its results are undefined by specification since the input
          // is an unsigned type, CL_UNSIGNED_INT16
          
            sad4x4 += abs((unsigned short)((read_imageui(img_ref, texSampler, (int2)(search_off_x + x, search_off_y + y) )).x) -
                  frame[cur_o + y * img_width + x]);
                  
          }
        }

        // Save this value into the local SAD array 
        blk_sad[search_pos] = sad4x4;
      }
    }

}
