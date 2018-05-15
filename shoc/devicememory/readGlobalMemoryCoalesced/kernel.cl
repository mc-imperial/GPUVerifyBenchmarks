//pass
//--global_size=10240 --local_size=256

//providence: ./DeviceMemory

__kernel void readGlobalMemoryCoalesced(__global float *data, __global float *output, int size)
{
    __requires(size == 16777216);

    int gid = get_global_id(0), num_thr = get_global_size(0), grpid=get_group_id(0), j = 0;
    float sum = 0;
    int s = gid;
    for (j=0 ; j<1024 ; ++j) {
       float a0 = data[(s+0)&(size-1)];
       float a1 = data[(s+10240)&(size-1)];
       float a2 = data[(s+20480)&(size-1)];
       float a3 = data[(s+30720)&(size-1)];
       float a4 = data[(s+40960)&(size-1)];
       float a5 = data[(s+51200)&(size-1)];
       float a6 = data[(s+61440)&(size-1)];
       float a7 = data[(s+71680)&(size-1)];
       float a8 = data[(s+81920)&(size-1)];
       float a9 = data[(s+92160)&(size-1)];
       float a10 = data[(s+102400)&(size-1)];
       float a11 = data[(s+112640)&(size-1)];
       float a12 = data[(s+122880)&(size-1)];
       float a13 = data[(s+133120)&(size-1)];
       float a14 = data[(s+143360)&(size-1)];
       float a15 = data[(s+153600)&(size-1)];
       sum += a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
       s = (s+163840)&(size-1);
    }
    output[gid] = sum;
}
