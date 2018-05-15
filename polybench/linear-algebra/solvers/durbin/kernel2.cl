//PASS
//--local_size=[32] --num_groups=[32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *y, __global double *z, int n, long c0)
{
  __requires(n == 1024);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (2))) & ((n) >= (c0)));
      for (long c1 = 32 * b0; c1 < c0 - 1; c1 += 1048576)
        if (c0 >= t0 + c1 + 2) {
          // shared
          y[t0 + c1] = z[t0 + c1];
        }
      __function_wide_invariant(__write_implies(y, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((((__write_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1))) >= (0))) & ((c0) >= ((((__write_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1))) + (2)))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1)))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(z, ((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1999))) & ((((__read_offset_bytes(z)) / (sizeof(double))) % (__ite((n) <= (2000), (n) - (1), 2000))) == (((32) * (b0)) + (t0)))) & ((c0) >= ((((32) * (b0)) + (t0)) + (2)))));
    }
}
