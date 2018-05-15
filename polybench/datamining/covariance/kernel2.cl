//PASS
//--local_size=[32,16] --num_groups=[32,32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *cov, int m, int n)
{
  __requires(m == 1024);
  __requires(n == 512);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires(((((m) >= (1)) & ((m) <= (2147483647))) & ((n) <= (2147483647))) & ((n) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < m; c0 += 8192)
        for (long c1 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8160) / 8192); c1 < m; c1 += 8192) {
          // shared
          for (long c3 = max(t1, t1 + 16 * floord(t0 - t1 + c0 - c1 - 1, 16) + 16); c3 <= min(31, m - c1 - 1); c3 += 16)
            cov[(t0 + c0) * m + (c1 + c3)] = 0.;
        }
      __function_wide_invariant(__write_implies(cov, (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)) >= (0))) & ((m) >= ((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) + (1)))) & ((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) >= ((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m)))) & (((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(cov)) / (sizeof(double))) % (m)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(cov)) / (sizeof(double))) / (m)) % (m))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(cov)) / (sizeof(double))) % (m))) % (16)) == (0))));
    }
}
