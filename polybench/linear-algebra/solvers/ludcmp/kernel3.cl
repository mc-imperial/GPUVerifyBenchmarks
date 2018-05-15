//PASS
//--local_size=[32] --num_groups=[8]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel3(__global double *A, __global double *w, int n, long c0)
{
  __requires(n == 256);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    double private_A_0[1][1];
    double private_w;

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((n) >= ((c0) + (1)))) & ((c0) >= (0)));
      for (long c1 = 32 * b0 + 1048576 * ((-32 * b0 + c0 + 1048544) / 1048576); c1 < n; c1 += 1048576)
        if (n >= t0 + c1 + 1 && t0 + c1 >= c0) {
          private_A_0[0][0] = A[c0 * n + (t0 + c1)];
          // shared
          {
            private_w = private_A_0[0][0];
            for (long c3 = 1; c3 <= c0; c3 += 1)
              private_w -= (A[c0 * n + (c3 - 1)] * A[(c3 - 1) * n + (t0 + c1)]);
            private_A_0[0][0] = private_w;
          }
          A[c0 * n + (t0 + c1)] = private_A_0[0][0];
        }
      __function_wide_invariant(__write_implies(A, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) >= ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((c0) == ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & ((c0) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((n) >= (((((((32) * (b0)) + (t0)) + ((1048575) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) % (1048576)) + ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) + (1)))) | ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((c0) >= ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (c0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) % (1048576)) == (0)))));
    }
}
