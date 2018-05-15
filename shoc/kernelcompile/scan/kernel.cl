//pass
//--num_groups=1 --local_size=32

__kernel void
scan(__global float *g_odata,
     __global float *g_idata,
     __global float *g_blockSums,
     int n,
     int blockIndex,
     int baseIndex,
     int storeSum,
     __local float *s_data)
{
    int ai, bi;
    int mem_ai, mem_bi;
    int bIndex;

    // load data into shared memory
    if (baseIndex == 0)
    {
        bIndex = get_group_id(0) * (get_local_size(0) << 1);
    }
    else
    {
        bIndex = baseIndex;
    }

    int thid = get_local_id(0);
    mem_ai = bIndex + thid;
    mem_bi = mem_ai + get_local_size(0);

    ai = thid;
    bi = thid + get_local_size(0);

    // Cache the computational window in shared memory
    // pad values beyond n with zeros
    s_data[ai] = g_idata[mem_ai];
    if (bi < n)
    {
        s_data[bi] = g_idata[mem_bi];
    }
    else
    {
        s_data[bi] = 0.0f;
    }

    unsigned int stride = 1;

    // build the sum in place up the tree
    for (int d = get_local_size(0);
         __invariant(__implies((d == 0) & __write(s_data), thid == 0)),
         __invariant(__implies((d == 0) & __read(s_data), thid == 0)),
         d > 0; d >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thid < d)
        {
            int i  = 2 * stride * thid;
            int aii = i + stride - 1;
            int bii = aii + stride;

            s_data[bii] += s_data[aii];
        }
        stride *= 2;
    }

    bIndex = (blockIndex == 0) ? get_group_id(0) : blockIndex;

    if (get_local_id(0) == 0)
    {
        int index = (get_local_size(0) << 1) - 1;

        if (storeSum == 1)
        {
            // write this block's total sum to the corresponding
            // index in the blockSums array
            g_blockSums[bIndex] = s_data[index];
        }

        // zero the last element in the scan so it will propagate
        // back to the front
        s_data[index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // traverse down the tree building the scan in place
    for (int d = 1; d <= get_local_size(0); d *= 2)
    {
        stride >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (thid < d)
        {
            int i  = 2 * stride * thid;
            int aii = i + stride - 1;
            int bii = aii + stride;

            float t  = s_data[aii];
            s_data[aii] = s_data[bii];
            s_data[bii] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // write results to global memory
    g_odata[mem_ai] = s_data[ai];
    if (bi < n)
    {
        g_odata[mem_bi] = s_data[bi];
    }
}
