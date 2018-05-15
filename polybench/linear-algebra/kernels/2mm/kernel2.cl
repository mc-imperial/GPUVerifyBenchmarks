//PASS
//--local_size=[32,16] --num_groups=[32,32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *C, __global double *D, __global double *tmp, int nj, int nl, int nk, int ni)
{
  __requires(nj == 1024);
  __requires(nl == 1024);
  __requires(nk == 1024);
  __requires(ni == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_C[32][32];
    double private_D[1][2];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((((((nj) >= (1)) & ((nj) <= (2147483647))) & ((nl) >= (1))) & ((nl) <= (2147483647))) & ((nk) >= (0))) & ((nk) <= (2147483647))) & ((ni) >= (1))) & ((ni) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < ni; c0 += 8192)
        for (long c1 = 32 * b1; c1 < nl; c1 += 8192) {
          if (ni >= t0 + c0 + 1 && nl >= t1 + c1 + 1) {
            private_D[0][0] = D[(t0 + c0) * nl + (t1 + c1)];
            if (nl >= t1 + c1 + 17)
              private_D[0][1] = D[(t0 + c0) * nl + (t1 + c1 + 16)];
          }
          for (long c2 = 0; c2 < nj; c2 += 32) {
            if (nj >= t0 + c2 + 1)
              for (long c4 = t1; c4 <= min(31, nl - c1 - 1); c4 += 16)
                shared_C[t0][c4] = C[(t0 + c2) * nl + (c1 + c4)];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (ni >= t0 + c0 + 1 && nl >= t1 + c1 + 1) {
              // shared
              for (long c3 = 0; c3 <= min(31, nj - c2 - 1); c3 += 1) {
                private_D[0][0] += (tmp[(t0 + c0) * nj + (c2 + c3)] * shared_C[c3][t1]);
                if (nl >= t1 + c1 + 17)
                  private_D[0][1] += (tmp[(t0 + c0) * nj + (c2 + c3)] * shared_C[c3][t1 + 16]);
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          }
          if (ni >= t0 + c0 + 1 && nl >= t1 + c1 + 1) {
            D[(t0 + c0) * nl + (t1 + c1)] = private_D[0][0];
            if (nl >= t1 + c1 + 17)
              D[(t0 + c0) * nl + (t1 + c1 + 16)] = private_D[0][1];
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(D, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((ni) >= (((((__write_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni)) + (1)))) & (((((__write_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni)) >= (0))) & ((nl) >= ((((__write_offset_bytes(D)) / (sizeof(double))) % (nl)) + (1)))) & ((((__write_offset_bytes(D)) / (sizeof(double))) % (nl)) >= (0))) & (((((__write_offset_bytes(D)) / (sizeof(double))) % (nl)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(D)) / (sizeof(double))) % (nl)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(D)) / (sizeof(double))) % (nl))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(D, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((ni) >= (((((__read_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni)) + (1)))) & (((((__read_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni)) >= (0))) & ((nl) >= ((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) + (1)))) & ((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) >= (0))) & (((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(D)) / (sizeof(double))) % (nl))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(tmp, (((((((((32) * (b1)) >= ((t1) + (1))) & ((nl) >= ((((32) * (b1)) + (t1)) + (1)))) & ((ni) >= (((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni)) + (1)))) & (((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni)) >= (0))) & ((nj) >= ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj)) + (1)))) & ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj)) >= (0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni))) % (8192)) == (0))) | ((((((((b1) == (0)) & ((nl) >= ((t1) + (1)))) & ((ni) >= (((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni)) + (1)))) & (((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni)) >= (0))) & ((nj) >= ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj)) + (1)))) & ((((__read_offset_bytes(tmp)) / (sizeof(double))) % (nj)) >= (0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(tmp)) / (sizeof(double))) / (nj)) % (ni))) % (8192)) == (0)))));
      __function_wide_invariant(__read_implies(C, (((((((nj) >= (((((__read_offset_bytes(C)) / (sizeof(double))) / (nl)) % (nj)) + (1))) & (((((__read_offset_bytes(C)) / (sizeof(double))) / (nl)) % (nj)) >= (0))) & ((nl) >= ((((__read_offset_bytes(C)) / (sizeof(double))) % (nl)) + (1)))) & ((((__read_offset_bytes(C)) / (sizeof(double))) % (nl)) >= (0))) & (((((((32) * (b1)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(C)) / (sizeof(double))) % (nl)))) + (31)) % (8192)) <= (31))) & ((((t0) - ((((__read_offset_bytes(C)) / (sizeof(double))) / (nl)) % (nj))) % (32)) == (0))) & ((((t1) - (((__read_offset_bytes(C)) / (sizeof(double))) % (nl))) % (16)) == (0))));
    }
}
