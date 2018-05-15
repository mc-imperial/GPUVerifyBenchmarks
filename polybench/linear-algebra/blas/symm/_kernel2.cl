//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *A, __global double *B, __global double *C, double alpha, double beta, __global double *temp2, int n, int m, long c0, long c1)
{
  __requires(n == 1024);
  __requires(m == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((n) <= (2147483647)) & ((m) <= (2147483647))) & ((m) >= ((c0) + (1)))) & ((c0) >= (0))) & ((n) >= ((c1) + (1)))) & ((c1) >= (0)));
      // shared
      C[c0 * n + c1] = (((beta * C[c0 * n + c1]) + ((alpha * B[c0 * n + c1]) * A[c0 * m + c0])) + (alpha * temp2[0]));
      __function_wide_invariant(__write_implies(C, ((c0) == ((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (m))) & ((c1) == (((__write_offset_bytes(C)) / (sizeof(double))) % (n)))));
      __function_wide_invariant(__read_implies(A, ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m))) & ((c0) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m)))));
      __function_wide_invariant(__read_implies(C, ((c0) == ((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (m))) & ((c1) == (((__read_offset_bytes(C)) / (sizeof(double))) % (n)))));
      __function_wide_invariant(__read_implies(B, ((c0) == ((((__read_offset_bytes(B)) / (sizeof(double))) / (n)) % (m))) & ((c1) == (((__read_offset_bytes(B)) / (sizeof(double))) % (n)))));
    }
}
