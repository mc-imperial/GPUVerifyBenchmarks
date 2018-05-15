//pass
//--local_size=[512] --num_groups=[16385]

//Also run with --local_size=[512] --num_groups=[41]

__kernel void uniformAdd(unsigned int n, __global unsigned int *dataBase, unsigned int data_offset, __global unsigned int *interBase, unsigned int inter_offset)
{
    __local unsigned int uni;
    
    __global unsigned int *data = dataBase + data_offset;
    __global unsigned int *inter = interBase + inter_offset;
       
    if (get_local_id(0) == 0) { uni = inter[get_group_id(0)]; }
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    unsigned int g_ai = get_group_id(0)*2*get_local_size(0) + get_local_id(0);
    unsigned int g_bi = g_ai + get_local_size(0);

    if (g_ai < n) { data[g_ai] += uni; }
    if (g_bi < n) { data[g_bi] += uni; }
}
