//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel7(__global double *alpha, __global double *beta, __global double *r, __global double *sum, int n, long c0)
{
  __requires(n == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (1))) & ((n) >= ((c0) + (1))));
      // shared
      alpha[0] = ((-(r[c0] + sum[0])) / beta[0]);
      __function_wide_invariant(__read_implies(r, (c0) == (((__read_offset_bytes(r)) / (sizeof(double))) % (__ite((n) >= (2), n, 1)))));
    }
}
