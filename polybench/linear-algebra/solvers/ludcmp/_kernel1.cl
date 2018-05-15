//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *A, __global double *w, int n, long c0, long c1, long c2)
{
  __requires(n == 256);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((((n) <= (2147483647)) & ((n) >= ((c0) + (1)))) & ((c0) >= ((c1) + (1)))) & ((c1) >= ((c2) + (1)))) & ((c2) >= (0)));
      // shared
      w[0] -= (A[c0 * n + c2] * A[c2 * n + c1]);
      __function_wide_invariant(__read_implies(A, (((c1) == (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) & ((c2) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) | (((c0) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) & ((c2) == (((__read_offset_bytes(A)) / (sizeof(double))) % (n))))));
    }
}
