//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel7(__global double *w, __global double *y, int n, long c0)
{
  __requires(n == 256);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & (((n) + (c0)) >= (1))) & ((c0) <= (0)));
      // shared
      w[0] = y[-c0];
      __function_wide_invariant(__read_implies(y, ((((__read_offset_bytes(y)) / (sizeof(double))) % (n)) + (c0)) == (0)));
    }
}
