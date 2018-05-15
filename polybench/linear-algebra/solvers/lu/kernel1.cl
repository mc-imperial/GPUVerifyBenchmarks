//PASS
//--local_size=[32,16] --num_groups=[2,2]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *A, int n, long c0)
{
  __requires(n == 64);
  __requires(c0 < 2 * n - 2); // Possible overflow in __requires
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_A_0[32][32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires((((n) <= (2147483647)) & (((2) * (n)) >= ((c0) + (3)))) & ((c0) >= (1)));
      for (long c1 = max(32 * b0, 32 * b0 + 8192 * floord(-n - 32 * b0 + c0 - 30, 8192) + 8192);
           __global_invariant(__read_implies(shared_A_0, __read_offset_bytes(shared_A_0)/sizeof(double)/32 == t0)),
           __global_invariant(__read_implies(shared_A_0, __read_offset_bytes(shared_A_0)/sizeof(double)%32%16 == t1)),
           c1 <= min(n - 1, c0); c1 += 8192) {
        for (long c2 = max(32 * b1, 32 * b1 + 8192 * floord(-n - 32 * b1 + c0 - 31, 8192) + 8192);
             __global_invariant(__read_implies(shared_A_0, __read_offset_bytes(shared_A_0)/sizeof(double)/32 == t0)),
             __global_invariant(__read_implies(shared_A_0, __read_offset_bytes(shared_A_0)/sizeof(double)%32%16 == t1)),
             c2 <= min(min(c0 - c1, c1 + 30), (c0 + 1) / 2 - 1); c2 += 8192) {
          if (n >= t0 + c1 + 1)
            for (long c4 = max(t1, ((t1 + c0 - c2 + 17) % 16) - c0 + c2 + 31);
                 c4 <= min(31, n - c0 + c2 + 30); c4 += 16)
              shared_A_0[t0][c4] = A[(t0 + c1) * n + (c0 - c2 + c4 - 31)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          // shared
          for (long c4 = max(t1, t1 + 16 * floord(-n - t1 + c0 - c2, 16) + 16);
               __global_invariant(__read_implies(shared_A_0, __read_offset_bytes(shared_A_0)/sizeof(double)/32 == t0)),
               __global_invariant(__implies(__dominator_enabled() &
                                            __read(shared_A_0), __read_offset_bytes(shared_A_0)/sizeof(double)%32%16 == -(t1 + 16) + 31)),
               __global_invariant(__write_implies(shared_A_0, __write_offset_bytes(shared_A_0)/sizeof(double)/32 == t0)),
               __global_invariant(__write_implies(shared_A_0, __write_offset_bytes(shared_A_0)/sizeof(double)%32%16 == -(t1 + 16) + 31)),
               c4 <= min(min(31, t0 + c1 - c2 - 1), -t0 + c0 - c1 - c2); c4 += 16)
            shared_A_0[t0][-c4 + 31] -= (A[(t0 + c1) * n + (c2 + c4)] * A[(c2 + c4) * n + (c0 - c2 - c4)]);
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          for (long c4 = max(max(t1, t1 + 16 * floord(t0 - t1 - c0 + c1 + c2 - 2, 16) + 48), ((t0 + t1 + c1 - c2 + 16) % 16) - t0 - c1 + c2 + 32);
               __global_invariant(__read_implies(shared_A_0, __read_offset_bytes(shared_A_0)/sizeof(double)/32 == t0)),
               __global_invariant(__read_implies(shared_A_0, __read_offset_bytes(shared_A_0)/sizeof(double)%32%16 == t1)),
               c4 <= min(31, n - c0 + c2 + 30); c4 += 16)
            A[(t0 + c1) * n + (c0 - c2 + c4 - 31)] = shared_A_0[t0][c4];
        }
      }
      __function_wide_invariant(__write_implies(A, (((((((n) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) + (1))) & ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) >= ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) >= ((c0) + (1)))) & ((c0) >= (((__write_offset_bytes(A)) / (sizeof(double))) % (n)))) & (((((((-32) * (b1)) - (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) + (c0)) + (8160)) % (8192)) >= (8160))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & (((((((15) * (t1)) + (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) - (c0)) + (15)) % (16)) == (0))));
      __function_wide_invariant(__read_implies(A, (((((((((((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (0)) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((c0) == (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))))) & ((((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) % (8192)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (((((((((32) * (b0)) + (t0)) + ((7680) * (t1))) + ((511) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) + (8191)) % (8192)) + ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) + (1)))) & ((((t1) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (16)) == (0))) | (((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & (((n) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) >= ((c0) + (1)))) & ((c0) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))))) & (((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) % (16)) == (0)))) | ((((((((((((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & ((c0) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (31)))) & ((((n) + ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) + (29)) >= ((t0) + (c0)))) & (((((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (31)) >= ((t0) + (c0)))) & (((((((-32) * (b1)) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (c0)) + (8160)) % (8192)) >= (8160))) & (((((((((-32) * (b1)) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (c0)) + (8160)) % (8192)) + (t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (8160)))) & ((((2) * ((((((-32) * (b1)) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (c0)) + (8160)) % (8192))) + ((2) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n)))) >= ((c0) + (16321)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & (((((((15) * (t1)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) - (c0)) + (15)) % (16)) == (0)))) | (((((((((((b1) == (0)) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (t0))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & ((c0) >= (((__read_offset_bytes(A)) / (sizeof(double))) % (n)))) & (((t0) + (c0)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (30)) >= (c0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & (((((((15) * (t1)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) - (c0)) + (15)) % (16)) == (0)))));
    }
}
