//pass
//--num_groups=2 --local_size=2

__kernel void
uniformAdd(__global float *g_data,
           __global float *uniforms,
           int n,
           int blockOffset,
           int baseIndex)
{
    float uni = 0.0f;

    uni = uniforms[get_group_id(0) + blockOffset];
    unsigned int address = (get_group_id(0) * (get_local_size(0) << 1)) +
                           baseIndex + get_local_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);

    g_data[address] += uni;
    if (get_local_id(0) + get_local_size(0) < n)
    {
        g_data[address + get_local_size(0)] +=  uni;
    }
}
