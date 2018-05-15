//PASS
//--local_size=[32,16] --num_groups=[32,32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel5(__global double *R, int n, int m)
{
  __requires(n == 1024);
  __requires(m == 256);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires(((((n) >= (2)) & ((n) <= (2147483647))) & ((m) <= (2147483647))) & ((m) >= (-2147483648)));
      for (long c1 = 32 * b0; c1 < n - 1; c1 += 8192)
        for (long c2 = 32 * b1 + 8192 * ((-32 * b1 + c1 + 8161) / 8192); c2 < n; c2 += 8192) {
          // shared
          for (long c4 = max(t1, t1 + 16 * floord(t0 - t1 + c1 - c2, 16) + 16); c4 <= min(31, n - c2 - 1); c4 += 16)
            R[(t0 + c1) * n + (c2 + c4)] = 0;
        }
      __function_wide_invariant(__write_implies(R, (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__write_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((((__write_offset_bytes(R)) / (sizeof(double))) % (n)) >= (((((__write_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((n) >= ((((__write_offset_bytes(R)) / (sizeof(double))) % (n)) + (1)))) & (((((__write_offset_bytes(R)) / (sizeof(double))) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(R)) / (sizeof(double))) % (n)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(R)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(R)) / (sizeof(double))) % (n))) % (16)) == (0))));
    }
}
