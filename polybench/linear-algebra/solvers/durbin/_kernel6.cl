//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel6(__global double *r, __global double *sum, __global double *y, int n, long c0, long c1)
{
  __requires(n == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((n) <= (2147483647)) & ((n) >= ((c0) + (1)))) & ((c0) >= ((c1) + (1)))) & ((c1) >= (0)));
      // shared
      sum[0] += (r[c0 - c1 - 1] * y[c1]);
      __function_wide_invariant(__read_implies(r, (((((__read_offset_bytes(r)) / (sizeof(double))) % (__ite((n) >= (2), n, 1))) + (c1)) + (1)) == (c0)));
      __function_wide_invariant(__read_implies(y, (c1) == (((__read_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1)))));
    }
}
