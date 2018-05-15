//PASS
//--local_size=[32] --num_groups=[2]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, int n)
{
  __requires(n == 64);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((n) >= (1)) & ((n) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 1048576)
        if (n >= t0 + c0 + 1) {
          // shared
          A[(t0 + c0) * n + (t0 + c0)] = A[(t0 + c0) * n + (t0 + c0)];
        }
      __function_wide_invariant(__write_implies(A, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (1048576)) == (0))));
    }
}
