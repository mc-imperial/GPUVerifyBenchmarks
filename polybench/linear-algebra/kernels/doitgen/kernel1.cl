//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *A, __global double *C4, __global double *sum, int np, int nq, int nr, long c0, long c1)
{
  __requires(np == 512);
  __requires(nq == 512);
  __requires(nr == 512);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_A[1][1][32];
    double private_sum[1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((((((np) >= (1)) & ((np) <= (2147483647))) & ((nq) <= (2147483647))) & ((nr) <= (2147483647))) & ((nr) >= ((c0) + (1)))) & ((c0) >= (0))) & ((nq) >= ((c1) + (1)))) & ((c1) >= (0)));
      for (long c2 = 32 * b0; c2 < np; c2 += 1048576) {
        if (np >= t0 + c2 + 1)
          private_sum[0] = sum[t0 + c2];
        for (long c3 = 0; c3 < np; c3 += 32) {
          if (np >= t0 + c3 + 1)
            shared_A[0][0][t0] = A[(c0 * nq + c1) * np + (t0 + c3)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (np >= t0 + c2 + 1) {
            // shared
            for (long c5 = 0; c5 <= min(31, np - c3 - 1); c5 += 1)
              private_sum[0] += (shared_A[0][0][c5] * C4[(c3 + c5) * np + (t0 + c2)]);
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (np >= t0 + c2 + 1)
          sum[t0 + c2] = private_sum[0];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
      __function_wide_invariant(__write_implies(sum, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((np) >= ((((__write_offset_bytes(sum)) / (sizeof(double))) % (np)) + (1)))) & ((((__write_offset_bytes(sum)) / (sizeof(double))) % (np)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(sum)) / (sizeof(double))) % (np))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(sum, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((np) >= ((((__read_offset_bytes(sum)) / (sizeof(double))) % (np)) + (1)))) & ((((__read_offset_bytes(sum)) / (sizeof(double))) % (np)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(sum)) / (sizeof(double))) % (np))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(C4, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((np) >= (((((__read_offset_bytes(C4)) / (sizeof(double))) / (np)) % (np)) + (1)))) & (((((__read_offset_bytes(C4)) / (sizeof(double))) / (np)) % (np)) >= (0))) & ((np) >= ((((__read_offset_bytes(C4)) / (sizeof(double))) % (np)) + (1)))) & ((((__read_offset_bytes(C4)) / (sizeof(double))) % (np)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(C4)) / (sizeof(double))) % (np))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, (((((np) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (np)) + (1))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (np)) >= (0))) & ((c0) == (((((__read_offset_bytes(A)) / (sizeof(double))) / (np)) / (nq)) % (nr)))) & ((c1) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (np)) % (nq)))) & ((((t0) - (((__read_offset_bytes(A)) / (sizeof(double))) % (np))) % (32)) == (0))));
    }
}
