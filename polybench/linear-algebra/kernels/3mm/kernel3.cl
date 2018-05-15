//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel3(__global double *C, __global double *D, __global double *E, __global double *F, __global double *G, int nl, int nj, int nm, int nk, int ni)
{
  __requires(nl == 512);
  __requires(nj == 512);
  __requires(nm == 512);
  __requires(nk == 512);
  __requires(ni == 512);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_F[32][32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((((((((nl) >= (1)) & ((nl) <= (2147483647))) & ((nj) >= (1))) & ((nj) <= (2147483647))) & ((nm) >= (0))) & ((nm) <= (2147483647))) & ((nk) >= (0))) & ((nk) <= (2147483647))) & ((ni) <= (2147483647))) & ((ni) >= (-2147483648)));
      if (ni >= 1) {
        for (long c0 = 32 * b0; c0 < nl; c0 += 1048576)
          for (long c1 = 0; c1 < nj; c1 += 32) {
            if (nl >= t0 + c0 + 1)
              for (long c2 = 0; c2 <= min(31, nj - c1 - 1); c2 += 1)
                shared_F[c2][t0] = F[(c1 + c2) * nl + (t0 + c0)];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (nl >= t0 + c0 + 1) {
              // shared
              for (long c3 = 0; c3 <= min(31, nj - c1 - 1); c3 += 1) {
                for (long c4 = 0; c4 < nm; c4 += 1)
                  shared_F[c3][t0] += (C[(c1 + c3) * nm + c4] * D[c4 * nl + (t0 + c0)]);
                for (long c4 = 0; c4 < ni; c4 += 1)
                  G[c4 * nl + (t0 + c0)] += (E[c4 * nj + (c1 + c3)] * shared_F[c3][t0]);
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (nm >= 1 && nl >= t0 + c0 + 1)
              for (long c2 = 0; c2 <= min(31, nj - c1 - 1); c2 += 1)
                F[(c1 + c2) * nl + (t0 + c0)] = shared_F[c2][t0];
          }
      } else if (nm >= 1)
        for (long c0 = 32 * b0; c0 < nl; c0 += 1048576)
          for (long c1 = 0; c1 < nj; c1 += 32) {
            if (nl >= t0 + c0 + 1)
              for (long c2 = 0; c2 <= min(31, nj - c1 - 1); c2 += 1)
                shared_F[c2][t0] = F[(c1 + c2) * nl + (t0 + c0)];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (nl >= t0 + c0 + 1) {
              // shared
              for (long c3 = 0; c3 <= min(31, nj - c1 - 1); c3 += 1)
                for (long c4 = 0; c4 < nm; c4 += 1)
                  shared_F[c3][t0] += (C[(c1 + c3) * nm + c4] * D[c4 * nl + (t0 + c0)]);
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            if (nl >= t0 + c0 + 1)
              for (long c2 = 0; c2 <= min(31, nj - c1 - 1); c2 += 1)
                F[(c1 + c2) * nl + (t0 + c0)] = shared_F[c2][t0];
          }
      __function_wide_invariant(__write_implies(G, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((ni) >= (((((__write_offset_bytes(G)) / (sizeof(double))) / (nl)) % (ni)) + (1)))) & (((((__write_offset_bytes(G)) / (sizeof(double))) / (nl)) % (ni)) >= (0))) & ((nl) >= ((((__write_offset_bytes(G)) / (sizeof(double))) % (nl)) + (1)))) & ((((__write_offset_bytes(G)) / (sizeof(double))) % (nl)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(G)) / (sizeof(double))) % (nl))) % (1048576)) == (0))));
      __function_wide_invariant(__write_implies(F, ((((((nm) >= (1)) & ((nj) >= (((((__write_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj)) + (1)))) & (((((__write_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj)) >= (0))) & ((nl) >= ((((__write_offset_bytes(F)) / (sizeof(double))) % (nl)) + (1)))) & ((((__write_offset_bytes(F)) / (sizeof(double))) % (nl)) >= (t0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(F)) / (sizeof(double))) % (nl))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(G, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((ni) >= (((((__read_offset_bytes(G)) / (sizeof(double))) / (nl)) % (ni)) + (1)))) & (((((__read_offset_bytes(G)) / (sizeof(double))) / (nl)) % (ni)) >= (0))) & ((nl) >= ((((__read_offset_bytes(G)) / (sizeof(double))) % (nl)) + (1)))) & ((((__read_offset_bytes(G)) / (sizeof(double))) % (nl)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(G)) / (sizeof(double))) % (nl))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(E, (((((nl) >= ((((32) * (b0)) + (t0)) + (1))) & ((ni) >= (((((__read_offset_bytes(E)) / (sizeof(double))) / (nj)) % (ni)) + (1)))) & (((((__read_offset_bytes(E)) / (sizeof(double))) / (nj)) % (ni)) >= (0))) & ((nj) >= ((((__read_offset_bytes(E)) / (sizeof(double))) % (nj)) + (1)))) & ((((__read_offset_bytes(E)) / (sizeof(double))) % (nj)) >= (0))));
      __function_wide_invariant(__read_implies(D, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((nm) >= (((((__read_offset_bytes(D)) / (sizeof(double))) / (nl)) % (nm)) + (1)))) & (((((__read_offset_bytes(D)) / (sizeof(double))) / (nl)) % (nm)) >= (0))) & ((nl) >= ((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) + (1)))) & ((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(D)) / (sizeof(double))) % (nl))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(F, ((((((((nm) >= (1)) & ((ni) <= (0))) & ((nj) >= (((((__read_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj)) + (1)))) & (((((__read_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj)) >= (0))) & ((nl) >= ((((__read_offset_bytes(F)) / (sizeof(double))) % (nl)) + (1)))) & ((((__read_offset_bytes(F)) / (sizeof(double))) % (nl)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(F)) / (sizeof(double))) % (nl))) % (1048576)) == (0))) | (((((((ni) >= (1)) & ((nj) >= (((((__read_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj)) + (1)))) & (((((__read_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj)) >= (0))) & ((nl) >= ((((__read_offset_bytes(F)) / (sizeof(double))) % (nl)) + (1)))) & ((((__read_offset_bytes(F)) / (sizeof(double))) % (nl)) >= (t0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(F)) / (sizeof(double))) % (nl))) % (1048576)) == (0)))));
      __function_wide_invariant(__read_implies(C, (((((nl) >= ((((32) * (b0)) + (t0)) + (1))) & ((nj) >= (((((__read_offset_bytes(C)) / (sizeof(double))) / (nm)) % (nj)) + (1)))) & (((((__read_offset_bytes(C)) / (sizeof(double))) / (nm)) % (nj)) >= (0))) & ((nm) >= ((((__read_offset_bytes(C)) / (sizeof(double))) % (nm)) + (1)))) & ((((__read_offset_bytes(C)) / (sizeof(double))) % (nm)) >= (0))));
    }
}
