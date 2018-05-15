//PASS
//--local_size=[32] --num_groups=[32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel21(__global double *p, int n, int tsteps, long c0)
{
  __requires(n == 1024);
  __requires(tsteps == 512);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((((n) >= (3)) & ((n) <= (2147483647))) & ((tsteps) <= (2147483647))) & ((c0) >= (1))) & ((tsteps) >= (c0)));
      for (long c1 = 32 * b0; c1 < n - 1; c1 += 1048576)
        if (n >= t0 + c1 + 2 && t0 + c1 >= 1) {
          // shared
          p[(t0 + c1) * n + 0] = 0;
        }
      __function_wide_invariant(__write_implies(p, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__write_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1))) + (2)))) & (((((__write_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1))) >= (1))) & ((((__write_offset_bytes(p)) / (sizeof(double))) % (n)) == (0))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1)))) % (1048576)) == (0))));
    }
}
