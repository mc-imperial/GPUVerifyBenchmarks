//PASS
//--local_size=[32] --num_groups=[8]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel6(__global double *A, __global double *Q, __global double *R, int n, int m, long c0)
{
  __requires(n == 1024);
  __requires(m == 256);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_R[1][1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((n) <= (2147483647)) & ((m) >= (1))) & ((m) <= (2147483647))) & (((5) * (n)) >= ((c0) + (2)))) & ((c0) >= (0))) & ((((c0) - (3)) % (5)) == (0)));
      if (t0 == 0)
        shared_R[0][0] = R[((c0 - 3) / 5) * n + ((c0 - 3) / 5)];
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      for (long c1 = 32 * b0; c1 < m; c1 += 1048576)
        if (m >= t0 + c1 + 1) {
          // shared
          Q[(t0 + c1) * n + ((c0 - 3) / 5)] = (A[(t0 + c1) * n + ((c0 - 3) / 5)] / shared_R[0][0]);
        }
      __function_wide_invariant(__write_implies(Q, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= (((((__write_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__write_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((c0) == (((5) * (((__write_offset_bytes(Q)) / (sizeof(double))) % (n))) + (3)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((c0) == (((5) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (3)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(R, (((t0) == (0)) & ((((__read_offset_bytes(R)) / (sizeof(double))) % (n)) == ((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)))) & ((c0) == (((5) * ((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n))) + (3)))));
    }
}
