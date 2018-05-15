//PASS
//--local_size=[32] --num_groups=[32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel9(__global double *alpha, __global double *y, __global double *z, int n, long c0)
{
  __requires(n == 1024);
  __requires(c0 <= n); // Possible overflow in __requires
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_alpha;
    __local double shared_y_1[32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (1))) & ((n) >= ((c0) + (1))));
      if (t0 == 0 && c0 >= 32 * b0 + 1)
        shared_alpha = *alpha;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      for (long c1 = 32 * b0; c1 < c0; c1 += 1048576) {
        if (t0 + c0 >= c1 + 32)
          shared_y_1[t0] = y[t0 + c0 - c1 - 32];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (c0 >= t0 + c1 + 1) {
          // shared
          z[t0 + c1] = (y[t0 + c1] + (shared_alpha * shared_y_1[-t0 + 31]));
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
      __function_wide_invariant(__write_implies(z, ((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1999))) & ((((__write_offset_bytes(z)) / (sizeof(double))) % (__ite((n) <= (2000), (n) - (1), 2000))) == (((32) * (b0)) + (t0)))) & ((c0) >= ((((32) * (b0)) + (t0)) + (1)))));
      __function_wide_invariant(__read_implies(y, ((((((__read_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1))) >= (0)) & (((t0) + (c0)) >= ((((__read_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1))) + (32)))) & ((((((((32) * (b0)) + ((1048575) * (t0))) + (((__read_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1)))) - (c0)) + (32)) % (1048576)) == (0))) | ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((((__read_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1))) >= (0))) & ((c0) >= ((((__read_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1))) + (1)))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(y)) / (sizeof(double))) % (__ite((n) >= (2), n, 1)))) % (1048576)) == (0)))));
    }
}
