//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, __global double *nrm, int n, int m, long c0, long c1)
{
  __requires(n == 1024);
  __requires(m == 256);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((((((n) <= (2147483647)) & ((m) <= (2147483647))) & (((5) * (n)) >= ((c0) + (4)))) & ((c0) >= (1))) & ((m) >= ((c1) + (1)))) & ((c1) >= (0))) & ((((c0) - (1)) % (5)) == (0)));
      // shared
      nrm[0] += (A[c1 * n + ((c0 - 1) / 5)] * A[c1 * n + ((c0 - 1) / 5)]);
      __function_wide_invariant(__read_implies(A, ((c0) == (((5) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (1))) & ((c1) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)))));
    }
}
