//pass
//--global_size=256 --local_size=256

#include "../common.h"

// This single group kernel takes the per block histograms
// from the reduction and performs an exclusive scan on them.
__kernel void
top_scan(__global FPTYPE * isums, 
         const int n,
         __local FPTYPE * lmem)
{
    __requires(n == 64);

    __local int s_seed;
    s_seed = 0; barrier(CLK_LOCAL_MEM_FENCE);
    
    // Decide if this is the last thread that needs to 
    // propagate the seed value
    int last_thread = (get_local_id(0) < n &&
                      (get_local_id(0)+1) == n) ? 1 : 0;

    for (int d = 0;
         __invariant(__implies(get_local_id(0) >= n, !__write(isums))),
         __invariant(__implies(get_local_id(0) >= n, !__read(isums))),
         d < 16; d++)
    {
        FPTYPE val = 0;
        // Load each block's count for digit d
        if (get_local_id(0) < n)
        {
            val = isums[(n * d) + get_local_id(0)];
        }
        // Exclusive scan the counts in local memory
        FPTYPE res = scanLocalMem(val, lmem, 1);
        // Write scanned value out to global
        if (get_local_id(0) < n)
        {
            isums[(n * d) + get_local_id(0)] = res + s_seed;
        }
#ifndef KERNEL_BUG
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        if (last_thread) 
        {
            s_seed += res + val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
