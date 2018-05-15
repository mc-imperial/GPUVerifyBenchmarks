//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *A, __global double *B, __global double *temp2, int n, int m, long c0, long c1, long c2)
{
  __requires(n == 1024);
  __requires(m == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((((((n) <= (2147483647)) & ((m) <= (2147483647))) & ((m) >= ((c0) + (1)))) & ((n) >= ((c1) + (1)))) & ((c1) >= (0))) & ((c0) >= ((c2) + (1)))) & ((c2) >= (0)));
      // shared
      temp2[0] += (B[c2 * n + c1] * A[c0 * m + c2]);
      __function_wide_invariant(__read_implies(B, ((c1) == (((__read_offset_bytes(B)) / (sizeof(double))) % (n))) & ((c2) == ((((__read_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)))));
      __function_wide_invariant(__read_implies(A, ((c0) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m))) & ((c2) == (((__read_offset_bytes(A)) / (sizeof(double))) % (m)))));
    }
}
