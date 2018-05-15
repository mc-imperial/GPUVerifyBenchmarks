//PASS
//--local_size=[32] --num_groups=[64]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *A, __global double *tmp, __global double *y, int n, int m)
{
  __requires(n == 2048);
  __requires(m == 2048);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_tmp[32];
    double private_y[1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((n) >= (1)) & ((n) <= (2147483647))) & ((m) >= (1))) & ((m) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 1048576) {
        if (n >= t0 + c0 + 1)
          private_y[0] = y[t0 + c0];
        for (long c1 = 0; c1 < m; c1 += 32) {
          if (m >= t0 + c1 + 1)
            shared_tmp[t0] = tmp[t0 + c1];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (n >= t0 + c0 + 1) {
            // shared
            for (long c3 = 0; c3 <= min(31, m - c1 - 1); c3 += 1)
              private_y[0] = (private_y[0] + (A[(c1 + c3) * n + (t0 + c0)] * shared_tmp[c3]));
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (n >= t0 + c0 + 1)
          y[t0 + c0] = private_y[0];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
      __function_wide_invariant(__write_implies(y, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__write_offset_bytes(y)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(y)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(y)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(y, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__read_offset_bytes(y)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(y)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(y)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(tmp, (((m) >= ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (m)) + (1))) & ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (m)) >= (0))) & ((((t0) - (((__read_offset_bytes(tmp)) / (sizeof(double))) % (m))) % (32)) == (0))));
    }
}
