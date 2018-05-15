//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel8(__global double *A, __global double *w, __global double *x, int n, long c0, long c1)
{
  __requires(n == 256);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((n) <= (2147483647)) & ((c0) <= (0))) & (((c0) + (c1)) >= (1))) & ((n) >= ((c1) + (1))));
      // shared
      w[0] -= (A[-c0 * n + c1] * x[c1]);
      __function_wide_invariant(__read_implies(A, ((((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (c0)) == (0)) & ((c1) == (((__read_offset_bytes(A)) / (sizeof(double))) % (n)))));
      __function_wide_invariant(__read_implies(x, (c1) == (((__read_offset_bytes(x)) / (sizeof(double))) % (n))));
    }
}
