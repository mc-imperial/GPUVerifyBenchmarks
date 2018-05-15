//PASS
//--local_size=[32,16] --num_groups=[32,32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, __global double *C, double alpha, double beta, int n, int m)
{
  __requires(n == 1024);
  __requires(m == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_A_0[32][32];
    double private_C[1][2];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((n) >= (1)) & ((n) <= (2147483647))) & ((m) >= (0))) & ((m) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 8192)
        for (long c1 = 32 * b1; c1 <= min(n - 1, c0 + 31); c1 += 8192) {
          if (n >= t0 + c0 + 1 && t0 + c0 >= t1 + c1) {
            private_C[0][0] = C[(t0 + c0) * n + (t1 + c1)];
            if (t0 + c0 >= t1 + c1 + 16)
              private_C[0][1] = C[(t0 + c0) * n + (t1 + c1 + 16)];
          }
          for (long c2 = 0; c2 < m; c2 += 32) {
            if (n >= t0 + c0 + 1)
              for (long c4 = t1; c4 <= min(31, m - c2 - 1); c4 += 16)
                shared_A_0[t0][c4] = A[(t0 + c0) * m + (c2 + c4)];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (n >= t0 + c0 + 1 && t0 + c0 >= t1 + c1) {
              // shared
              {
                if (c2 == 0) {
                  private_C[0][0] *= beta;
                  if (t0 + c0 >= t1 + c1 + 16)
                    private_C[0][1] *= beta;
                }
                for (long c3 = 0; c3 <= min(31, m - c2 - 1); c3 += 1) {
                  private_C[0][0] += ((alpha * shared_A_0[t0][c3]) * A[(t1 + c1) * m + (c2 + c3)]);
                  if (t0 + c0 >= t1 + c1 + 16)
                    private_C[0][1] += ((alpha * shared_A_0[t0][c3]) * A[(t1 + c1 + 16) * m + (c2 + c3)]);
                }
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          if (m == 0) {
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (n >= t0 + c0 + 1 && t0 + c0 >= t1 + c1) {
              // shared
              {
                private_C[0][0] *= beta;
                if (t0 + c0 >= t1 + c1 + 16)
                  private_C[0][1] *= beta;
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          if (n >= t0 + c0 + 1 && t0 + c0 >= t1 + c1) {
            C[(t0 + c0) * n + (t1 + c1)] = private_C[0][0];
            if (t0 + c0 >= t1 + c1 + 16)
              C[(t0 + c0) * n + (t1 + c1 + 16)] = private_C[0][1];
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(C, (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((((__write_offset_bytes(C)) / (sizeof(double))) % (n)) >= (0))) & (((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (n)) >= (((__write_offset_bytes(C)) / (sizeof(double))) % (n)))) & (((((__write_offset_bytes(C)) / (sizeof(double))) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(C)) / (sizeof(double))) % (n)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(C)) / (sizeof(double))) % (n))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)) >= (0)) & ((m) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) >= (0))) & ((((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)) % (8192)))) & (((n) + ((8192) * (__ite(((((((32) * (b0)) + (t0)) + ((8192) * (t1))) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)))<0), -((-(((((32) * (b0)) + (t0)) + ((8192) * (t1))) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)))+(8192)-1)/(8192)), (((((32) * (b0)) + (t0)) + ((8192) * (t1))) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)))/(8192))))) >= (((((32) * (b0)) + (t0)) + ((8192) * (t1))) + (1)))) & ((((t1) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n))) % (16)) == (0))) | (((((((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)) + (1))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)) >= (((32) * (b1)) + (t0)))) & ((m) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) >= (0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(A)) / (sizeof(double))) % (m))) % (16)) == (0)))) | ((((((((((b1) == (b0)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) >= (0))) & ((((t0) - (t1)) % (16)) == (0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (n))) % (8192)) == (0)))));
      __function_wide_invariant(__read_implies(C, (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((((__read_offset_bytes(C)) / (sizeof(double))) % (n)) >= (0))) & (((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (n)) >= (((__read_offset_bytes(C)) / (sizeof(double))) % (n)))) & (((((__read_offset_bytes(C)) / (sizeof(double))) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(C)) / (sizeof(double))) % (n)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(C)) / (sizeof(double))) % (n))) % (16)) == (0))));
    }
}
