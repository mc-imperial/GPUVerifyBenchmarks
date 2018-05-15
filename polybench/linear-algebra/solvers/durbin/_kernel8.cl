//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel8(__global double *alpha, __global double *y, int n, long c0)
{
  __requires(n == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (1))) & ((n) >= ((c0) + (1))));
      // shared
      y[c0] = alpha[0];
      __function_wide_invariant(__write_implies(y, (c0) == (((__write_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1)))));
    }
}
