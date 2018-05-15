//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel8(__global double *corr, int m, int n)
{
  __requires(m == 512);
  __requires(n == 1024);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((m) >= (2)) & ((m) <= (2147483647))) & ((n) <= (2147483647))) & ((n) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < m - 1; c0 += 1048576)
        if (m >= t0 + c0 + 2) {
          // shared
          corr[(t0 + c0) * (m >= 2 ? m : 1) + (t0 + c0)] = 1.;
        }
      __function_wide_invariant(__write_implies(corr, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= (((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) + (2)))) & (((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) >= (0))) & ((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) == ((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1)))) % (1048576)) == (0))));
    }
}
