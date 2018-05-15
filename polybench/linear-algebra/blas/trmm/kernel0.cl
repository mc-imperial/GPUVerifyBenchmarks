//PASS
//--local_size=[32] --num_groups=[32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, __global double *B, double alpha, int n, int m)
{
  __requires(n == 1024);
  __requires(m == 1024);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_A[32][32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires(((((n) >= (1)) & ((n) <= (2147483647))) & ((m) >= (1))) & ((m) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 1048576)
        for (long c1 = 0; c1 <= m; c1 += 32) {
          if (m >= c1 + 1) {
            for (long c2 = 0; c2 <= min(m - 2, c1 + 30); c2 += 32) {
              if (m >= t0 + c2 + 1)
                for (long c3 = 0; c3 <= min(31, m - c1 - 1); c3 += 1)
                  shared_A[c3][t0] = A[(c1 + c3) * m + (t0 + c2)];
              barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
              if (n >= t0 + c0 + 1) {
                // shared
                {
                  for (long c4 = max(0, -c1 + c2 + 1); c4 <= min(31, m - c1 - 1); c4 += 1)
                    for (long c5 = 0; c5 <= min(31, c1 - c2 + c4 - 1); c5 += 1)
                      B[(c2 + c5) * n + (t0 + c0)] += (shared_A[c4][c5] * B[(c1 + c4) * n + (t0 + c0)]);
                  if (c1 + 31 >= m)
                    for (long c5 = 0; c5 <= min(31, m - c2 - 1); c5 += 1)
                      B[(c2 + c5) * n + (t0 + c0)] = (alpha * B[(c2 + c5) * n + (t0 + c0)]);
                }
              }
              barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            }
            if (m >= 33 && c1 + 1 == m && (m - 1) % 32 == 0) {
              if (t0 == 0)
                shared_A[0][0] = A[(m - 1) * m + (m - 1)];
              barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
              if (n >= t0 + c0 + 1) {
                // shared
                B[(m - 1) * n + (t0 + c0)] = (alpha * B[(m - 1) * n + (t0 + c0)]);
              }
              barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            }
            if (m == 1 && c1 == 0) {
              barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
              if (n >= t0 + c0 + 1) {
                // shared
                B[0 * n + (t0 + c0)] = (alpha * B[0 * n + (t0 + c0)]);
              }
              barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            }
          } else
            for (long c2 = 0; c2 < m; c2 += 32) {
              barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
              if (n >= t0 + c0 + 1) {
                // shared
                for (long c5 = 0; c5 <= 31; c5 += 1)
                  B[(c2 + c5) * n + (t0 + c0)] = (alpha * B[(c2 + c5) * n + (t0 + c0)]);
              }
              barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            }
        }
      __function_wide_invariant(__write_implies(B, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= (((((__write_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__write_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((n) >= ((((__write_offset_bytes(B)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(B)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(B)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(B, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= (((((__read_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__read_offset_bytes(B)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((n) >= ((((__read_offset_bytes(B)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(B)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(B)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, (((((((m) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m)) + (1))) & (((m) + (t0)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) + (2)))) & ((m) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) >= (0))) & (((t0) + ((32) * (__ite((((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m))<0), -((-((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m))+(32)-1)/(32)), ((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m))/(32))))) >= (((__read_offset_bytes(A)) / (sizeof(double))) % (m)))) & ((((t0) - (((__read_offset_bytes(A)) / (sizeof(double))) % (m))) % (32)) == (0))) | ((((((m) >= (33)) & ((t0) == (0))) & ((((((__read_offset_bytes(A)) / (sizeof(double))) / (m)) % (m)) + (1)) == (m))) & (((((__read_offset_bytes(A)) / (sizeof(double))) % (m)) + (1)) == (m))) & ((((m) - (1)) % (32)) == (0)))));
    }
}
