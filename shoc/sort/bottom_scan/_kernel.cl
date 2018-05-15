//pass
//--global_size=16384 --local_size=256 

#include "../common.h"

__kernel void 
bottom_scan(__global const FPTYPE * in,
            __global const FPTYPE * isums,
            __global FPTYPE * out,
            const int n,
            __local FPTYPE * lmem,
            const int shift)
{
    __requires(n == 262144);

    // Use local memory to cache the scanned seeds
    __local FPTYPE l_scanned_seeds[16];
    
    // Keep a shared histogram of all instances seen by the current
    // block
    __local FPTYPE l_block_counts[16];
    
    // Keep a private histogram as well
    __private int histogram[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0  };

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

    // Set the histogram in local memory to zero
    // and read in the scanned seeds from gmem
    if (get_local_id(0) < 16)
    {
        l_block_counts[get_local_id(0)] = 0;
        l_scanned_seeds[get_local_id(0)] = 
            isums[(get_local_id(0)*get_num_groups(0))+get_group_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Scan multiple elements per thread
    while (window < block_stop)
    {
        // Reset histogram
        for (int q = 0; q < 16; q++) histogram[q] = 0;
        FPVECTYPE val_4;
        FPVECTYPE key_4;        

        if (i < block_stop) // Make sure we don't read out of bounds
        {
            val_4 = in4[i];
            
            // Mask the keys to get the appropriate digit
            key_4.x = (val_4.x >> shift) & 0xFU;
            key_4.y = (val_4.y >> shift) & 0xFU;
            key_4.z = (val_4.z >> shift) & 0xFU;
            key_4.w = (val_4.w >> shift) & 0xFU;
            
            // Update the histogram
            histogram[key_4.x]++;
            histogram[key_4.y]++;
            histogram[key_4.z]++;
            histogram[key_4.w]++;
        } 
                
        // Scan the digit counts in local memory
        for (int digit = 0; digit < 16; digit++)
        {
            histogram[digit] = scanLocalMem(histogram[digit], lmem, 1);
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i < block_stop) // Make sure we don't write out of bounds
        {
            int address;
            address = histogram[key_4.x] + l_scanned_seeds[key_4.x] + l_block_counts[key_4.x];
            out[address] = val_4.x;
            histogram[key_4.x]++;
            
            address = histogram[key_4.y] + l_scanned_seeds[key_4.y] + l_block_counts[key_4.y];
            out[address] = val_4.y;
            histogram[key_4.y]++;
            
            address = histogram[key_4.z] + l_scanned_seeds[key_4.z] + l_block_counts[key_4.z];
            out[address] = val_4.z;
            histogram[key_4.z]++;
            
            address = histogram[key_4.w] + l_scanned_seeds[key_4.w] + l_block_counts[key_4.w];
            out[address] = val_4.w;
            histogram[key_4.w]++;
        }
                
        // Before proceeding, make sure everyone has finished their current
        // indexing computations.
        barrier(CLK_LOCAL_MEM_FENCE);
        // Now update the seed array.
        if (get_local_id(0) == get_local_size(0)-1)
        {
            for (int q = 0; q < 16; q++)
            {
                 l_block_counts[q] += histogram[q];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Advance window
        window += get_local_size(0);
        i += get_local_size(0);
    }
}
