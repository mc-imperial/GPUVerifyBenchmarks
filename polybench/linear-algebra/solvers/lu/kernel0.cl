//PASS
//--local_size=[32] --num_groups=[2]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, int n, long c0)
{
  __requires(n == 64);
  __requires(c0 < 2 * n - 2); // Possible overflow in __requires
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_A_1[1][1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((n) <= (2147483647)) & (((2) * (n)) >= ((c0) + (3)))) & ((c0) >= (0))) & (((c0) % (2)) == (0)));
      if (t0 == 0)
        shared_A_1[0][0] = A[(c0 / 2) * n + (c0 / 2)];
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      for (long c1 = 32 * b0 + 1048576 * ((-64 * b0 + c0 + 2097091) / 2097152); c1 < n; c1 += 1048576)
        if (n >= t0 + c1 + 1 && 2 * t0 + 2 * c1 >= c0 + 2) {
          // shared
          A[(t0 + c1) * n + (c0 / 2)] /= shared_A_1[0][0];
        }
      __function_wide_invariant(__write_implies(A, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((c0) == ((2) * (((__write_offset_bytes(A)) / (sizeof(double))) % (n))))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((c0) == ((2) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n))))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (1048576)) == (0))) | ((((t0) == (0)) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((c0) == ((2) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))))));
    }
}
