//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel10(__global double *d, __global double *f, int n, int tsteps)
{
  __requires(n == 1024);
  __requires(tsteps == 512);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((n) >= (0)) & ((n) <= (2147483647))) & ((tsteps) <= (2147483647))) & ((tsteps) >= (-2147483648)));
      // shared
      f[0] = d[0];
    }
}
