//PASS
//--local_size=[32] --num_groups=[64]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *y, int n, int m)
{
  __requires(n == 2048);
  __requires(m == 2048);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((n) >= (1)) & ((n) <= (2147483647))) & ((m) <= (2147483647))) & ((m) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < n; c0 += 1048576)
        if (n >= t0 + c0 + 1) {
          // shared
          y[t0 + c0] = 0;
        }
      __function_wide_invariant(__write_implies(y, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__write_offset_bytes(y)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(y)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(y)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
    }
}
