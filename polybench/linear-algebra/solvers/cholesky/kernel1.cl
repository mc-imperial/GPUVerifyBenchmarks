//PASS
//--local_size=[32,16] --num_groups=[2,2]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *A, int n)
{
  __requires(n == 64);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((n) >= (2)) & ((n) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 8192)
        if (n >= t0 + c0 + 1)
          for (long c1 = 32 * b1; c1 <= min(n - 2, c0 + 30); c1 += 8192) {
            // shared
            for (long c3 = t1; c3 <= min(31, t0 + c0 - c1 - 1); c3 += 16)
              A[(t0 + c0) * n + (c1 + c3)] = A[(t0 + c0) * n + (c1 + c3)];
          }
      __function_wide_invariant(__write_implies(A, (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & (((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(A, (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & (((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) % (16)) == (0))));
    }
}
