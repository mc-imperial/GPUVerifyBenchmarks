//PASS
//--local_size=[32,16] --num_groups=[32,32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel3(__global double *A, __global double *B, __global double *C, double alpha, int n, int m)
{
  __requires(n == 1024);
  __requires(m == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_A[32][32];
    double private_C[2][1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires(((((n) >= (1)) & ((n) <= (2147483647))) & ((m) >= (2))) & ((m) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 8192)
        for (long c1 = 32 * b1; c1 < m - 1; c1 += 8192) {
          if (n >= t0 + c0 + 1 && m >= t1 + c1 + 2) {
            private_C[0][0] = C[(t1 + c1) * n + (t0 + c0)];
            if (m >= t1 + c1 + 18)
              private_C[1][0] = C[(t1 + c1 + 16) * n + (t0 + c0)];
          }
          for (long c2 = c1; c2 < m; c2 += 32) {
            if (m >= t0 + c2 + 1)
              for (long c4 = t1; c4 <= min(31, m - c1 - 1); c4 += 16)
                shared_A[t0][c4] = A[(t0 + c2) * m + (c1 + c4)];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (n >= t0 + c0 + 1) {
              // shared
              for (long c3 = max(0, t1 + c1 - c2 + 1); c3 <= min(31, m - c2 - 1); c3 += 1) {
                private_C[0][0] += ((alpha * B[(c2 + c3) * n + (t0 + c0)]) * shared_A[c3][t1]);
                if (c2 + c3 >= t1 + c1 + 17)
                  private_C[1][0] += ((alpha * B[(c2 + c3) * n + (t0 + c0)]) * shared_A[c3][t1 + 16]);
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          if (n >= t0 + c0 + 1 && m >= t1 + c1 + 2) {
            C[(t1 + c1) * n + (t0 + c0)] = private_C[0][0];
            if (m >= t1 + c1 + 18)
              C[(t1 + c1 + 16) * n + (t0 + c0)] = private_C[1][0];
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(C, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((m) >= (((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (m)) + (2)))) & (((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((n) >= ((((__write_offset_bytes(C)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(C)) / (sizeof(double))) % (n)) >= (0))) & ((((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= (((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (m)) % (8192)))) & ((((t1) - ((((__write_offset_bytes(C)) / (sizeof(double))) / (n)) % (m))) % (16)) == (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(C)) / (sizeof(double))) % (n))) % (8192)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((m) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m)) + (1))) & ((m) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) >= (0))) & ((((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m)) + (31)) >= ((t0) + (((__read_offset_bytes(A)) / (sizeof(double))) % (m))))) & (((m) + (29)) >= (((((((32) * (b1)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(A)) / (sizeof(double))) % (m)))) + (31)) % (8192)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (m))))) & (((((((32) * (b1)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(A)) / (sizeof(double))) % (m)))) + (31)) % (8192)) <= (31))) & ((((t0) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m))) % (32)) == (0))) & ((((t1) - (((__read_offset_bytes(A)) / (sizeof(double))) % (m))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(C, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((m) >= (((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (m)) + (2)))) & (((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((n) >= ((((__read_offset_bytes(C)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(C)) / (sizeof(double))) % (n)) >= (0))) & ((((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= (((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (m)) % (8192)))) & ((((t1) - ((((__read_offset_bytes(C)) / (sizeof(double))) / (n)) % (m))) % (16)) == (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(C)) / (sizeof(double))) % (n))) % (8192)) == (0))));
      __function_wide_invariant(__read_implies(B, ((((((((32) * (b1)) >= ((t1) + (1))) & (((((__read_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)) >= ((((32) * (b1)) + (t1)) + (1)))) & ((m) >= (((((__read_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)) + (1)))) & ((n) >= ((((__read_offset_bytes(B)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(B)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(B)) / (sizeof(double))) % (n))) % (8192)) == (0))) | (((((((b1) == (0)) & (((((__read_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)) >= ((t1) + (1)))) & ((m) >= (((((__read_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)) + (1)))) & ((n) >= ((((__read_offset_bytes(B)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(B)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(B)) / (sizeof(double))) % (n))) % (8192)) == (0)))));
    }
}
