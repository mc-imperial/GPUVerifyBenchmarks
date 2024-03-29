//PASS
//--local_size=[32,16] --num_groups=[16,16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *F, int nl, int nj, int nm, int nk, int ni)
{
  __requires(nl == 512);
  __requires(nj == 512);
  __requires(nm == 512);
  __requires(nk == 512);
  __requires(ni == 512);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((((((((nl) >= (1)) & ((nl) <= (2147483647))) & ((nj) >= (1))) & ((nj) <= (2147483647))) & ((nm) >= (0))) & ((nm) <= (2147483647))) & ((nk) >= (0))) & ((nk) <= (2147483647))) & ((ni) <= (2147483647))) & ((ni) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < nj; c0 += 8192)
        if (nj >= t0 + c0 + 1)
          for (long c1 = 32 * b1; c1 < nl; c1 += 8192) {
            // shared
            for (long c3 = t1; c3 <= min(31, nl - c1 - 1); c3 += 16)
              F[(t0 + c0) * nl + (c1 + c3)] = 0;
          }
      __function_wide_invariant(__write_implies(F, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((nj) >= (((((__write_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj)) + (1)))) & (((((__write_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj)) >= (0))) & ((nl) >= ((((__write_offset_bytes(F)) / (sizeof(double))) % (nl)) + (1)))) & ((((__write_offset_bytes(F)) / (sizeof(double))) % (nl)) >= (0))) & (((((__write_offset_bytes(F)) / (sizeof(double))) % (nl)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(F)) / (sizeof(double))) % (nl)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(F)) / (sizeof(double))) / (nl)) % (nj))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(F)) / (sizeof(double))) % (nl))) % (16)) == (0))));
    }
}
