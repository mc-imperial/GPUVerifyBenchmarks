//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel4(__global double *sum, int n, long c0)
{
  __requires(n == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (1))) & ((n) >= ((c0) + (1))));
      // shared
      sum[0] = 0.;
    }
}
