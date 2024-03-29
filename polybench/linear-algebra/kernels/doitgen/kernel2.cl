//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *A, __global double *sum, int np, int nq, int nr, long c0, long c1)
{
  __requires(np == 512);
  __requires(nq == 512);
  __requires(nr == 512);
  __requires(c0 <= nr); // Possible overflow in __requires
  __requires(c1 <= nq); // Possible overflow in __requires
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((((np) >= (1)) & ((np) <= (2147483647))) & ((nq) <= (2147483647))) & ((nr) <= (2147483647))) & ((nr) >= ((c0) + (1)))) & ((c0) >= (0))) & ((nq) >= ((c1) + (1)))) & ((c1) >= (0)));
      for (long c2 = 32 * b0; c2 < np; c2 += 1048576)
        if (np >= t0 + c2 + 1) {
          // shared
          A[(c0 * nq + c1) * np + (t0 + c2)] = sum[t0 + c2];
        }
      __function_wide_invariant(__write_implies(A, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((np) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (np)) + (1)))) & ((((__write_offset_bytes(A)) / (sizeof(double))) % (np)) >= (0))) & ((c0) == (((((__write_offset_bytes(A)) / (sizeof(double))) / (np)) / (nq)) % (nr)))) & ((c1) == ((((__write_offset_bytes(A)) / (sizeof(double))) / (np)) % (nq)))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(A)) / (sizeof(double))) % (np))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(sum, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((np) >= ((((__read_offset_bytes(sum)) / (sizeof(double))) % (np)) + (1)))) & ((((__read_offset_bytes(sum)) / (sizeof(double))) % (np)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(sum)) / (sizeof(double))) % (np))) % (1048576)) == (0))));
    }
}
