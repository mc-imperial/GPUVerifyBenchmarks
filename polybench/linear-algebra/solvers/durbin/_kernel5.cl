//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel5(__global double *beta, int n)
{
  __requires(n == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((n) <= (2147483647)) & ((n) >= (-2147483648)));
      // shared
      beta[0] = 1;
    }
}
