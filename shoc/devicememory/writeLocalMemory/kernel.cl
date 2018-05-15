//pass
//--global_size=10240 --local_size=256

__kernel void writeLocalMemory(__global float *output, int size)
{
    __requires(size == 16777216);

    int gid = get_global_id(0), num_thr = get_global_size(0), grpid=get_group_id(0), j = 0;
    float sum = 0;
    int tid=get_local_id(0), localSize=get_local_size(0), litems=4096/localSize, goffset=localSize*grpid+tid*litems;
    int s = tid;
    __local float lbuf[4096];
    for (j=0 ; j<3000 ; ++j) {
       lbuf[(s+0)&(4095)] = gid;
       lbuf[(s+1)&(4095)] = gid;
       lbuf[(s+2)&(4095)] = gid;
       lbuf[(s+3)&(4095)] = gid;
       lbuf[(s+4)&(4095)] = gid;
       lbuf[(s+5)&(4095)] = gid;
       lbuf[(s+6)&(4095)] = gid;
       lbuf[(s+7)&(4095)] = gid;
       lbuf[(s+8)&(4095)] = gid;
       lbuf[(s+9)&(4095)] = gid;
       lbuf[(s+10)&(4095)] = gid;
       lbuf[(s+11)&(4095)] = gid;
       lbuf[(s+12)&(4095)] = gid;
       lbuf[(s+13)&(4095)] = gid;
       lbuf[(s+14)&(4095)] = gid;
       lbuf[(s+15)&(4095)] = gid;
       s = (s+16)&(4095);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (j=0 ; j<litems ; ++j)
       output[gid] = lbuf[tid];
}
