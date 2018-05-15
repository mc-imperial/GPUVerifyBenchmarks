//PASS
//--local_size=[32,16] --num_groups=[32,16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel5(__global double *data, double float_n, __global double *stddev, int m, int n)
{
  __requires(m == 512);
  __requires(n == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_stddev[32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((m) >= (1)) & ((m) <= (2147483647))) & ((n) >= (1))) & ((n) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 8192)
        for (long c1 = 32 * b1; c1 < m; c1 += 8192) {
          if (t0 == 0)
            for (long c2 = t1; c2 <= min(31, m - c1 - 1); c2 += 16)
              shared_stddev[c2] = stddev[c1 + c2];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (n >= t0 + c0 + 1) {
            // shared
            for (long c3 = t1; c3 <= min(31, m - c1 - 1); c3 += 16)
              data[(t0 + c0) * m + (c1 + c3)] /= (sqrt(float_n) * shared_stddev[c3]);
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(data, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__write_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1)))) & (((((__write_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__write_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & ((((__write_offset_bytes(data)) / (sizeof(double))) % (m)) >= (0))) & (((((__write_offset_bytes(data)) / (sizeof(double))) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(data)) / (sizeof(double))) % (m)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(data)) / (sizeof(double))) / (m)) % (n))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(data)) / (sizeof(double))) % (m))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(data, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1)))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= (0))) & (((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(stddev, (((((t0) == (0)) & ((m) >= ((((__read_offset_bytes(stddev)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(stddev)) / (sizeof(double))) % (m)) >= (0))) & (((((((32) * (b1)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(stddev)) / (sizeof(double))) % (m)))) + (31)) % (8192)) <= (31))) & ((((t1) - (((__read_offset_bytes(stddev)) / (sizeof(double))) % (m))) % (16)) == (0))));
    }
}
