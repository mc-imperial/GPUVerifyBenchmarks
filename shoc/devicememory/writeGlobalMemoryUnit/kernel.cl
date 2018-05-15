//pass
//--global_size=10240 --local_size=256

__kernel void writeGlobalMemoryUnit(__global float *output, int size)
{
    __requires(size == 16777216);

    int gid = get_global_id(0), num_thr = get_global_size(0), grpid=get_group_id(0), j = 0;
    float sum = 0;
    int s = gid*1024;
    for (j=0 ; j<512 ; ++j) {
       output[(s+0)&(size-1)] = gid;
       output[(s+1)&(size-1)] = gid;
       output[(s+2)&(size-1)] = gid;
       output[(s+3)&(size-1)] = gid;
       output[(s+4)&(size-1)] = gid;
       output[(s+5)&(size-1)] = gid;
       output[(s+6)&(size-1)] = gid;
       output[(s+7)&(size-1)] = gid;
       output[(s+8)&(size-1)] = gid;
       output[(s+9)&(size-1)] = gid;
       output[(s+10)&(size-1)] = gid;
       output[(s+11)&(size-1)] = gid;
       output[(s+12)&(size-1)] = gid;
       output[(s+13)&(size-1)] = gid;
       output[(s+14)&(size-1)] = gid;
       output[(s+15)&(size-1)] = gid;
       s = (s+16)&(size-1);
    }
}
