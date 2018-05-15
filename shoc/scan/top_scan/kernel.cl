//pass
//--global_size=256 --local_size=256

#include "../common.h"

__kernel void
top_scan(__global FPTYPE * isums, 
         const int n,
         __local FPTYPE * lmem)
{
    __requires(n == 64);

    FPTYPE val = 0.0f;    
    if (get_local_id(0) < n)
    {
        val = isums[get_local_id(0)];
    }

    val = scanLocalMem(val, lmem, 1);

    if (get_local_id(0) < n)
    {
        isums[get_local_id(0)] = val;
    }
}
