//PASS
//--local_size=[32,16] --num_groups=[16,16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel7(__global double *corr, int m, int n)
{
  __requires(m == 512);
  __requires(n == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_corr_0[32][32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires(((((m) >= (2)) & ((m) <= (2147483647))) & ((n) <= (2147483647))) & ((n) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < m - 1; c0 += 8192)
        for (long c1 = 32 * b1 + 8192 * ((-32 * b1 + c0 + 8161) / 8192); c1 < m; c1 += 8192) {
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          // shared
          for (long c3 = max(t1, t1 + 16 * floord(t0 - t1 + c0 - c1, 16) + 16); c3 <= min(31, m - c1 - 1); c3 += 16)
            shared_corr_0[c3][t0] = corr[(t0 + c0) * (m >= 2 ? m : 1) + (c1 + c3)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (m >= t0 + c1 + 1)
            for (long c3 = t1; c3 <= min(31, t0 - c0 + c1 - 1); c3 += 16)
              corr[(t0 + c1) * (m >= 2 ? m : 1) + (c0 + c3)] = shared_corr_0[t0][c3];
        }
      __function_wide_invariant(__write_implies(corr, ((((((((b0) <= (254)) & ((m) >= (((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) + (1)))) & (((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) >= ((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) + (1)))) & ((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) >= (0))) & (((((((32) * (b0)) + ((7680) * (t1))) + ((511) * (((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))))) + (31)) % (8192)) <= (31))) & ((((((32) * (b1)) + (t0)) - ((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1)))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1)))) % (16)) == (0))) | ((((((((b0) == (255)) & ((m) >= (((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) + (1)))) & (((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) >= ((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) + (1)))) & ((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) >= (0))) & ((((((7680) * (t1)) + ((511) * (((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))))) + (8191)) % (8192)) <= (31))) & ((((((32) * (b1)) + (t0)) - ((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1)))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1)))) % (16)) == (0)))));
      __function_wide_invariant(__read_implies(corr, (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__read_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) >= (0))) & ((((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) >= (((((__read_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) + (1)))) & ((m) >= ((((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) + (1)))) & (((((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1)))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1)))) % (16)) == (0))));
    }
}
