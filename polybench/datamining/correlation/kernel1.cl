//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *stddev, int m, int n)
{
  __requires(m == 512);
  __requires(n == 1024);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((m) >= (1)) & ((m) <= (2147483647))) & ((n) <= (2147483647))) & ((n) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < m; c0 += 1048576)
        if (m >= t0 + c0 + 1) {
          // shared
          stddev[t0 + c0] = 0.;
        }
      __function_wide_invariant(__write_implies(stddev, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= ((((__write_offset_bytes(stddev)) / (sizeof(double))) % (m)) + (1)))) & ((((__write_offset_bytes(stddev)) / (sizeof(double))) % (m)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(stddev)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
    }
}
