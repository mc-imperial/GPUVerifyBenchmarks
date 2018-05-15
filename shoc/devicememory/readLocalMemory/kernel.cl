//pass
//--global_size=10240 --local_size=256

//providence: ./DeviceMemory

__kernel void readLocalMemory(__global const float *data, __global float *output, int size)
{
    __requires(size == 16777216);

    int gid = get_global_id(0), num_thr = get_global_size(0), grpid=get_group_id(0), j = 0;
    float sum = 0;
    int tid=get_local_id(0), localSize=get_local_size(0), litems=4096/localSize, goffset=localSize*grpid+tid*litems;
    int s = tid;
    __local float lbuf[4096];
    for ( ;
         __global_invariant(__write_implies(lbuf, __write_offset_bytes(lbuf)/sizeof(float) - tid*litems < litems)),
         __global_invariant(j >= 0),
         j<litems && j<(size-goffset) ; ++j)
       lbuf[tid*litems+j] = data[goffset+j];
    for (int i=0 ;
         __global_invariant(__write_implies(lbuf, __write_offset_bytes(lbuf)/sizeof(float) - tid*litems < litems)),
         j<litems ; ++j,++i)
       lbuf[tid*litems+j] = data[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (j=0 ; j<3000 ; ++j) {
       float a0 = lbuf[(s+0)&(4095)];
       float a1 = lbuf[(s+1)&(4095)];
       float a2 = lbuf[(s+2)&(4095)];
       float a3 = lbuf[(s+3)&(4095)];
       float a4 = lbuf[(s+4)&(4095)];
       float a5 = lbuf[(s+5)&(4095)];
       float a6 = lbuf[(s+6)&(4095)];
       float a7 = lbuf[(s+7)&(4095)];
       float a8 = lbuf[(s+8)&(4095)];
       float a9 = lbuf[(s+9)&(4095)];
       float a10 = lbuf[(s+10)&(4095)];
       float a11 = lbuf[(s+11)&(4095)];
       float a12 = lbuf[(s+12)&(4095)];
       float a13 = lbuf[(s+13)&(4095)];
       float a14 = lbuf[(s+14)&(4095)];
       float a15 = lbuf[(s+15)&(4095)];
       sum += a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
       s = (s+16)&(4095);
    }
    output[gid] = sum;
}
