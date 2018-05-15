//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *alpha, __global double *beta, int n, long c0)
{
  __requires(n == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (1))) & ((n) >= ((c0) + (1))));
      // shared
      beta[0] = ((1 - (alpha[0] * alpha[0])) * beta[0]);
    }
}
