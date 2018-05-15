//pass
//--global_size=10240 --local_size=256

__kernel void writeGlobalMemoryCoalesced(__global float *output, int size)
{
    __requires(size == 16777216);
    int gid = get_global_id(0), num_thr = get_global_size(0), grpid=get_group_id(0), j = 0;
    float sum = 0;
    int s = gid;
    for (j=0 ; j<1024 ; ++j) {
       output[(s+0)&(size-1)] = gid;
       output[(s+10240)&(size-1)] = gid;
       output[(s+20480)&(size-1)] = gid;
       output[(s+30720)&(size-1)] = gid;
       output[(s+40960)&(size-1)] = gid;
       output[(s+51200)&(size-1)] = gid;
       output[(s+61440)&(size-1)] = gid;
       output[(s+71680)&(size-1)] = gid;
       output[(s+81920)&(size-1)] = gid;
       output[(s+92160)&(size-1)] = gid;
       output[(s+102400)&(size-1)] = gid;
       output[(s+112640)&(size-1)] = gid;
       output[(s+122880)&(size-1)] = gid;
       output[(s+133120)&(size-1)] = gid;
       output[(s+143360)&(size-1)] = gid;
       output[(s+153600)&(size-1)] = gid;
       s = (s+163840)&(size-1);
    }
}
