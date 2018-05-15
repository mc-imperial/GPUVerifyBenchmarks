//PASS
//--local_size=[32,16] --num_groups=[32,32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, __global double *B, double alpha, __global double *tmp, int nj, int nl, int nk, int ni)
{
  __requires(nj == 1024);
  __requires(nl == 1024);
  __requires(nk == 1024);
  __requires(ni == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_A[32][32];
    double private_tmp[1][2];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((((((nj) >= (1)) & ((nj) <= (2147483647))) & ((nl) >= (0))) & ((nl) <= (2147483647))) & ((nk) >= (0))) & ((nk) <= (2147483647))) & ((ni) >= (1))) & ((ni) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < ni; c0 += 8192)
        for (long c1 = 32 * b1; c1 < nj; c1 += 8192) {
          for (long c2 = 0; c2 < nk; c2 += 32) {
            if (ni >= t0 + c0 + 1)
              for (long c4 = t1; c4 <= min(31, nk - c2 - 1); c4 += 16)
                shared_A[t0][c4] = A[(t0 + c0) * nk + (c2 + c4)];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1) {
              // shared
              {
                if (c2 == 0) {
                  private_tmp[0][0] = 0;
                  if (nj >= t1 + c1 + 17)
                    private_tmp[0][1] = 0;
                }
                for (long c3 = 0; c3 <= min(31, nk - c2 - 1); c3 += 1) {
                  private_tmp[0][0] += ((alpha * shared_A[t0][c3]) * B[(c2 + c3) * nj + (t1 + c1)]);
                  if (nj >= t1 + c1 + 17)
                    private_tmp[0][1] += ((alpha * shared_A[t0][c3]) * B[(c2 + c3) * nj + (t1 + c1 + 16)]);
                }
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          if (nk == 0) {
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1) {
              // shared
              {
                private_tmp[0][0] = 0;
                if (nj >= t1 + c1 + 17)
                  private_tmp[0][1] = 0;
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1) {
            tmp[(t0 + c0) * nj + (t1 + c1)] = private_tmp[0][0];
            if (nj >= t1 + c1 + 17)
              tmp[(t0 + c0) * nj + (t1 + c1 + 16)] = private_tmp[0][1];
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(tmp, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((ni) >= (((((__write_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni)) + (1)))) & (((((__write_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni)) >= (0))) & ((nj) >= ((((__write_offset_bytes(tmp)) / (sizeof(double))) % (nj)) + (1)))) & ((((__write_offset_bytes(tmp)) / (sizeof(double))) % (nj)) >= (0))) & (((((__write_offset_bytes(tmp)) / (sizeof(double))) % (nj)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(tmp)) / (sizeof(double))) % (nj)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(tmp)) / (sizeof(double))) % (nj))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(B, ((((((((ni) >= ((((32) * (b0)) + (t0)) + (1))) & ((nk) >= (((((__read_offset_bytes(B)) / (sizeof(double))) / (nj)) % (nk)) + (1)))) & (((((__read_offset_bytes(B)) / (sizeof(double))) / (nj)) % (nk)) >= (0))) & ((nj) >= ((((__read_offset_bytes(B)) / (sizeof(double))) % (nj)) + (1)))) & ((((__read_offset_bytes(B)) / (sizeof(double))) % (nj)) >= (0))) & (((((__read_offset_bytes(B)) / (sizeof(double))) % (nj)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(B)) / (sizeof(double))) % (nj)) % (8192)))) & ((((t1) - (((__read_offset_bytes(B)) / (sizeof(double))) % (nj))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((ni) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (nk)) % (ni)) + (1))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (nk)) % (ni)) >= (t0))) & ((nk) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (nk)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (nk)) >= (0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (nk)) % (ni))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(A)) / (sizeof(double))) % (nk))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(tmp, (((((((((((nk) >= (1)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (8191))) & ((ni) >= (((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni)) + (1)))) & (((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni)) >= (0))) & ((nj) >= ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj)) + (1)))) & ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj)) >= (0))) & (((((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj))) % (16)) == (0))));
    }
}
