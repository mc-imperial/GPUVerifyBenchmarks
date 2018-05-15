//pass
//--global_size=10240 --local_size=256

__kernel void readGlobalMemoryUnit(__global float *data, __global float *output, int size)
{
    __requires(size == 16777216);
    int gid = get_global_id(0), num_thr = get_global_size(0), grpid=get_group_id(0), j = 0;
    float sum = 0;
    int s = gid*1024;
    for (j=0 ; j<512 ; ++j) {
       float a0 = data[(s+0)&(size-1)];
       float a1 = data[(s+1)&(size-1)];
       float a2 = data[(s+2)&(size-1)];
       float a3 = data[(s+3)&(size-1)];
       float a4 = data[(s+4)&(size-1)];
       float a5 = data[(s+5)&(size-1)];
       float a6 = data[(s+6)&(size-1)];
       float a7 = data[(s+7)&(size-1)];
       float a8 = data[(s+8)&(size-1)];
       float a9 = data[(s+9)&(size-1)];
       float a10 = data[(s+10)&(size-1)];
       float a11 = data[(s+11)&(size-1)];
       float a12 = data[(s+12)&(size-1)];
       float a13 = data[(s+13)&(size-1)];
       float a14 = data[(s+14)&(size-1)];
       float a15 = data[(s+15)&(size-1)];
       sum += a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
       s = (s+16)&(size-1);
    }
    output[gid] = sum;
}
