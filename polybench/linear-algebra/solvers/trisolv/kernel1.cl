//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *b, __global double *x, int n)
{
  __requires(n == 512);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((n) >= (1)) & ((n) <= (2147483647)));
      for (long c1 = 32 * b0; c1 < n; c1 += 1048576)
        if (n >= t0 + c1 + 1) {
          // shared
          x[t0 + c1] = b[t0 + c1];
        }
      __function_wide_invariant(__write_implies(x, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__write_offset_bytes(x)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(x)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(x)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(b, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__read_offset_bytes(b)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(b)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(b)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
    }
}
