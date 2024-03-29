//PASS
//--local_size=[32,16] --num_groups=[32,32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *D, double beta, int nj, int nl, int nk, int ni)
{
  __requires(nj == 1024);
  __requires(nl == 1024);
  __requires(nk == 1024);
  __requires(ni == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((((((nj) >= (0)) & ((nj) <= (2147483647))) & ((nl) >= (1))) & ((nl) <= (2147483647))) & ((nk) >= (0))) & ((nk) <= (2147483647))) & ((ni) >= (1))) & ((ni) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < ni; c0 += 8192)
        if (ni >= t0 + c0 + 1)
          for (long c1 = 32 * b1; c1 < nl; c1 += 8192) {
            // shared
            for (long c3 = t1; c3 <= min(31, nl - c1 - 1); c3 += 16)
              D[(t0 + c0) * nl + (c1 + c3)] *= beta;
          }
      __function_wide_invariant(__write_implies(D, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((ni) >= (((((__write_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni)) + (1)))) & (((((__write_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni)) >= (0))) & ((nl) >= ((((__write_offset_bytes(D)) / (sizeof(double))) % (nl)) + (1)))) & ((((__write_offset_bytes(D)) / (sizeof(double))) % (nl)) >= (0))) & (((((__write_offset_bytes(D)) / (sizeof(double))) % (nl)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(D)) / (sizeof(double))) % (nl)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(D)) / (sizeof(double))) % (nl))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(D, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((ni) >= (((((__read_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni)) + (1)))) & (((((__read_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni)) >= (0))) & ((nl) >= ((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) + (1)))) & ((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) >= (0))) & (((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(D)) / (sizeof(double))) % (nl)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(D)) / (sizeof(double))) / (nl)) % (ni))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(D)) / (sizeof(double))) % (nl))) % (16)) == (0))));
    }
}
