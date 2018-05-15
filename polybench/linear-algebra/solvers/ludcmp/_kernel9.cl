//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel9(__global double *A, __global double *w, __global double *x, int n, long c0)
{
  __requires(n == 256);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & (((n) + (c0)) >= (1))) & ((c0) <= (0)));
      // shared
      x[-c0] = (w[0] / A[-c0 * n + -c0]);
      __function_wide_invariant(__write_implies(x, ((((__write_offset_bytes(x)) / (sizeof(double))) % (n)) + (c0)) == (0)));
      __function_wide_invariant(__read_implies(A, ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) & ((((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (c0)) == (0))));
    }
}
