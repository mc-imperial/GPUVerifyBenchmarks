//PASS
//--local_size=[32] --num_groups=[64]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, __global double *tmp, __global double *x, int n, int m)
{
  __requires(n == 2048);
  __requires(m == 2048);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_A[32][32];
    double private_tmp[1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((n) >= (0)) & ((n) <= (2147483647))) & ((m) >= (1))) & ((m) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < m; c0 += 1048576) {
        for (long c1 = 0; c1 < n; c1 += 32) {
          if (n >= t0 + c1 + 1)
            for (long c2 = 0; c2 <= min(31, m - c0 - 1); c2 += 1)
              shared_A[c2][t0] = A[(c0 + c2) * n + (t0 + c1)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (m >= t0 + c0 + 1) {
            // shared
            {
              if (c1 == 0)
                private_tmp[0] = 0;
              for (long c3 = 0; c3 <= min(31, n - c1 - 1); c3 += 1)
                private_tmp[0] = (private_tmp[0] + (shared_A[t0][c3] * x[c1 + c3]));
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (n == 0) {
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (m >= t0 + c0 + 1) {
            // shared
            private_tmp[0] = 0;
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (m >= t0 + c0 + 1)
          tmp[t0 + c0] = private_tmp[0];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
      __function_wide_invariant(__write_implies(tmp, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= ((((__write_offset_bytes(tmp)) / (sizeof(double))) % (m)) + (1)))) & ((((__write_offset_bytes(tmp)) / (sizeof(double))) % (m)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(tmp)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((m) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) + (1))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & (((b0) + ((32768) * (__ite((((((-32) * (b0)) + ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m))) + (1048544))<0), -((-((((-32) * (b0)) + ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m))) + (1048544))+(1048576)-1)/(1048576)), ((((-32) * (b0)) + ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m))) + (1048544))/(1048576))))) >= (0))) & ((((((-32) * (b0)) + ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m))) + (1048544)) % (1048576)) >= (1048544))) & ((((t0) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) % (32)) == (0))));
      __function_wide_invariant(__read_implies(x, (((m) >= ((((32) * (b0)) + (t0)) + (1))) & ((n) >= ((((__read_offset_bytes(x)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(x)) / (sizeof(double))) % (n)) >= (0))));
      __function_wide_invariant(__read_implies(tmp, ((((((n) >= (1)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (m)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(tmp)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
    }
}
