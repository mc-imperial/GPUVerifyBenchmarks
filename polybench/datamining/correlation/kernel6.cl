//PASS
//--local_size=[32,16] --num_groups=[16,16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel6(__global double *corr, __global double *data, int m, int n)
{
  __requires(m == 512);
  __requires(n == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    double private_corr[1][2];
    __local double shared_data_0[32][32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((m) >= (2)) & ((m) <= (2147483647))) & ((n) <= (2147483647))) & ((n) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < m - 1; c0 += 8192)
        for (long c1 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8161) / 8192); c1 < m; c1 += 8192) {
          for (long c2 = 0; c2 < n; c2 += 32) {
            if (n >= t0 + c2 + 1)
              for (long c4 = t1; c4 <= min(31, m - c0 - 1); c4 += 16)
                shared_data_0[t0][c4] = data[(t0 + c2) * m + (c0 + c4)];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (m >= t0 + c0 + 2 && m >= t1 + c1 + 1) {
              // shared
              {
                if (t1 + c1 + 15 >= t0 + c0 && c2 == 0) {
                  if (t1 + c1 >= t0 + c0 + 1)
                    private_corr[0][0] = 0.;
                  if (m >= t1 + c1 + 17)
                    private_corr[0][1] = 0.;
                }
                if (c1 + 30 >= ((15 * t0 + t1 + 15) % 16) + t0 + c0)
                  for (long c3 = 0; c3 <= min(31, n - c2 - 1); c3 += 1) {
                    if (t1 + c1 >= t0 + c0 + 1)
                      private_corr[0][0] += (shared_data_0[c3][t0] * data[(c2 + c3) * m + (t1 + c1)]);
                    if (m >= t1 + c1 + 17)
                      private_corr[0][1] += (shared_data_0[c3][t0] * data[(c2 + c3) * m + (t1 + c1 + 16)]);
                  }
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          if (n <= 0) {
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (m >= t0 + c0 + 2 && m >= t1 + c1 + 1 && t1 + c1 + 15 >= t0 + c0) {
              // shared
              {
                if (t1 + c1 >= t0 + c0 + 1)
                  private_corr[0][0] = 0.;
                if (m >= t1 + c1 + 17)
                  private_corr[0][1] = 0.;
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          if (m >= t0 + c0 + 2 && m >= t1 + c1 + 1 && t1 + c1 + 15 >= t0 + c0) {
            if (t1 + c1 >= t0 + c0 + 1)
              corr[(t0 + c0) * (m >= 2 ? m : 1) + (t1 + c1)] = private_corr[0][0];
            if (m >= t1 + c1 + 17)
              corr[(t0 + c0) * (m >= 2 ? m : 1) + (t1 + c1 + 16)] = private_corr[0][1];
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(corr, (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) >= (0))) & ((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) >= (((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) + (1)))) & ((m) >= ((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) + (1)))) & (((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1)))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1)))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(data, (((((((((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= ((((32) * (b0)) + (t0)) + (1)))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & (((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) % (8192)))) & ((((t1) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (16)) == (0))) | ((((((((((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= (32))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & (((m) + (30)) >= (((((((32) * (b1)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(data)) / (sizeof(double))) % (m)))) + (31)) % (8192)) + (((__read_offset_bytes(data)) / (sizeof(double))) % (m))))) & (((m) + (29)) >= (((((((32) * (b0)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(data)) / (sizeof(double))) % (m)))) + (31)) % (8192)) + (((__read_offset_bytes(data)) / (sizeof(double))) % (m))))) & (((((((32) * (b0)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(data)) / (sizeof(double))) % (m)))) + (31)) % (8192)) <= (31))) & ((((t0) - ((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n))) % (32)) == (0))) & ((((t1) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (16)) == (0)))) | (((((((((b0) == (0)) & ((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1)))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= (0))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) <= (31))) & ((((t0) - ((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n))) % (32)) == (0))) & ((((t1) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (16)) == (0)))));
      __function_wide_invariant(__read_implies(corr, ((((((((((n) >= (1)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__read_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) >= (0))) & ((((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) >= (((((__read_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) + (1)))) & ((m) >= ((((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) + (1)))) & (((((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1)))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1)))) % (16)) == (0))));
    }
}
