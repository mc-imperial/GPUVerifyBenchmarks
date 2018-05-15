//PASS
//--local_size=[32,16] --num_groups=[32,32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel3(__global double *cov, __global double *data, double float_n, int m, int n)
{
  __requires(m == 1024);
  __requires(n == 512);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_data_0[32][32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires(((((m) >= (1)) & ((m) <= (2147483647))) & ((n) <= (2147483647))) & ((n) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < m; c0 += 8192)
        for (long c1 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8160) / 8192); c1 < m; c1 += 8192) {
          for (long c2 = -((-n + 2147483648) % 32) - n; c2 < 0; c2 += 32) {
            if (n + t0 + c2 >= 0)
              for (long c4 = t1; c4 <= min(31, m - c0 - 1); c4 += 16)
                shared_data_0[t0][c4] = data[(n + t0 + c2) * m + (c0 + c4)];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            // shared
            for (long c4 = max(t1, t1 + 16 * floord(t0 - t1 + c0 - c1 - 1, 16) + 16); c4 <= min(31, m - c1 - 1); c4 += 16)
              for (long c5 = max(0, -n - c2); c5 <= 31; c5 += 1)
                cov[(t0 + c0) * m + (c1 + c4)] += (shared_data_0[c5][t0] * data[(n + c2 + c5) * m + (c1 + c4)]);
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          // shared
          for (long c4 = max(t1, t1 + 16 * floord(t0 - t1 + c0 - c1 - 1, 16) + 16); c4 <= min(31, m - c1 - 1); c4 += 16) {
            cov[(t0 + c0) * m + (c1 + c4)] /= (float_n - 1.);
            cov[(c1 + c4) * m + (t0 + c0)] = cov[(t0 + c0) * m + (c1 + c4)];
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(cov, (((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) >= (0))) & ((m) >= ((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) + (1)))) & ((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) >= ((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)))) & (((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(cov)) / (sizeof(double))) % (m))) % (16)) == (0))) | ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((m) >= (((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) + (1)))) & ((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) >= (0))) & (((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) >= (((__write_offset_bytes(cov)) / (sizeof(double))) % (m)))) & ((((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= (((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) % (8192)))) & ((((t1) - ((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m))) % (16)) == (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(cov)) / (sizeof(double))) % (m))) % (8192)) == (0)))) | (((((((((b1) == (b0)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (8191))) & ((m) >= (((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) + (1)))) & (((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) >= (0))) & ((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) == ((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)))) & ((((t0) - (t1)) % (16)) == (0))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m))) % (8192)) == (0)))));
      __function_wide_invariant(__read_implies(data, ((((((((((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= (((32) * (b0)) + (t0)))) & (((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) % (8192)))) & ((((t1) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (16)) == (0))) | ((((((((((b1) == (b0)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1)))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= (0))) & ((((t0) - (t1)) % (16)) == (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (8192)) == (0)))) | (((((((((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= (32))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & (((m) + (30)) >= (((((((32) * (b1)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(data)) / (sizeof(double))) % (m)))) + (31)) % (8192)) + (((__read_offset_bytes(data)) / (sizeof(double))) % (m))))) & (((((((32) * (b0)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(data)) / (sizeof(double))) % (m)))) + (31)) % (8192)) <= (31))) & (((((n) + (t0)) - ((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n))) % (32)) == (0))) & ((((t1) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (16)) == (0)))) | (((((((((b0) == (0)) & ((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1)))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= (0))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) <= (31))) & (((((n) + (t0)) - ((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n))) % (32)) == (0))) & ((((t1) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (16)) == (0)))));
      __function_wide_invariant(__read_implies(cov, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__read_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) >= (0))) & ((m) >= ((((__read_offset_bytes(cov)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(cov)) / (sizeof(double))) % (m)) >= ((((__read_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)))) & (((((__read_offset_bytes(cov)) / (sizeof(double))) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(cov)) / (sizeof(double))) % (m)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(cov)) / (sizeof(double))) % (m))) % (16)) == (0))) | (((((((((b1) == (b0)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (8191))) & ((m) >= (((((__read_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) + (1)))) & (((((__read_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) >= (0))) & ((((__read_offset_bytes(cov)) / (sizeof(double))) % (m)) == ((((__read_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)))) & ((((t0) - (t1)) % (16)) == (0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m))) % (8192)) == (0)))));
    }
}
