//pass
//--num_groups=64 --local_size=256

#include "../common.h"
  
__kernel void 
bottom_scan(__global const FPTYPE * in,
            __global const FPTYPE * isums,
            __global FPTYPE * out,
            const int n,
            __local FPTYPE * lmem)
{
    __requires(n  == 262144);
    __local FPTYPE s_seed;

    // Prepare for reading 4-element vectors
    // Assume n is divisible by 4
    __global FPVECTYPE *in4  = (__global FPVECTYPE*) in;
    __global FPVECTYPE *out4 = (__global FPVECTYPE*) out;
    int n4 = n / 4; //vector type is 4 wide
    
    int region_size = n4 / get_num_groups(0);
    int block_start = get_group_id(0) * region_size;
    // Give the last block any extra elements
    int block_stop  = (get_group_id(0) == get_num_groups(0) - 1) ? 
        n4 : block_start + region_size;

    // Calculate starting index for this thread/work item
    int i = block_start + get_local_id(0);
    int window = block_start;

    // Seed the bottom scan with the results from the top scan (i.e. load the per
    // block sums from the previous kernel)
    FPTYPE seed = isums[get_group_id(0)];

    // Scan multiple elements per thread
    while (__global_invariant(i == window + get_local_id(0)),
           __global_invariant(__write_implies(out4, __write_offset_bytes(out4)/sizeof(FPVECTYPE) >= block_start)),
           __global_invariant(__write_implies(out4, __write_offset_bytes(out4)/sizeof(FPVECTYPE) < block_stop)),
           window < block_stop)
    {
        FPVECTYPE val_4;
        if (i < block_stop) // Make sure we don't read out of bounds
        {
            val_4 = in4[i];
        } 
        else
        {
            val_4.x = 0.0f;
            val_4.y = 0.0f;
            val_4.z = 0.0f;
            val_4.w = 0.0f;
        }
        
        // Serial scan in registers
        val_4.y += val_4.x;
        val_4.z += val_4.y;
        val_4.w += val_4.z;
        
        // ExScan sums in shared memory
        FPTYPE res = scanLocalMem(val_4.w, lmem, 1);

        // Update and write out to global memory
        val_4.x += res + seed;
        val_4.y += res + seed;
        val_4.z += res + seed;
        val_4.w += res + seed;

        if (i < block_stop) // Make sure we don't write out of bounds
        {
            out4[i] = val_4;
        }
                
        // Next seed will be the last value
        // Last thread puts seed into smem.
        if (get_local_id(0) == get_local_size(0)-1) s_seed = val_4.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Broadcast seed to other threads
        seed = s_seed;

        // Advance window
        window += get_local_size(0);
        i += get_local_size(0);
    }
}
