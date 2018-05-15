//PASS
//--local_size=[32] --num_groups=[32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel22(__global double *p, __global double *q, __global double *v, int n, int tsteps, long c0)
{
  __requires(n == 1024);
  __requires(tsteps == 512);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_p[32][32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires((((((n) >= (3)) & ((n) <= (2147483647))) & ((tsteps) <= (2147483647))) & ((c0) >= (1))) & ((tsteps) >= (c0)));
      for (long c1 = 32 * b0; c1 < n - 1; c1 += 1048576)
        for (long c2 = ((n - 3) % 32) - n - 29; c2 < 0; c2 += 32) {
          if (n + c2 + 30 >= t0)
            for (long c3 = 0; c3 <= min(31, n - c1 - 2); c3 += 1)
              shared_p[c3][t0] = p[(c1 + c3) * n + (t0 - c2 - 31)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (n >= t0 + c1 + 2 && t0 + c1 >= 1) {
            // shared
            for (long c4 = max(0, -n - c2 + 2); c4 <= 31; c4 += 1)
              v[(-c2 - c4) * n + (t0 + c1)] = ((shared_p[t0][-c4 + 31] * v[(-c2 - c4 + 1) * n + (t0 + c1)]) + q[(t0 + c1) * n + (-c2 - c4)]);
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(v, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__write_offset_bytes(v)) / (sizeof(double))) / (n)) % (n)) + (2)))) & (((((__write_offset_bytes(v)) / (sizeof(double))) / (n)) % (n)) >= (1))) & ((n) >= ((((__write_offset_bytes(v)) / (sizeof(double))) % (n)) + (2)))) & ((((__write_offset_bytes(v)) / (sizeof(double))) % (n)) >= (1))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(v)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(p, (((((((n) >= (((((__read_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1))) + (2))) & (((n) + (t0)) >= ((((__read_offset_bytes(p)) / (sizeof(double))) % (n)) + (2)))) & ((((__read_offset_bytes(p)) / (sizeof(double))) % (n)) >= ((t0) + (1)))) & ((n) >= ((((__read_offset_bytes(p)) / (sizeof(double))) % (n)) + (1)))) & (((b0) + ((32768) * (__ite((((((-32) * (b0)) + ((((__read_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1)))) + (1048544))<0), -((-((((-32) * (b0)) + ((((__read_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1)))) + (1048544))+(1048576)-1)/(1048576)), ((((-32) * (b0)) + ((((__read_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1)))) + (1048544))/(1048576))))) >= (0))) & ((((((-32) * (b0)) + ((((__read_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1)))) + (1048544)) % (1048576)) >= (1048544))) & (((((t0) - (((__read_offset_bytes(p)) / (sizeof(double))) % (n))) + (1)) % (32)) == (0))));
      __function_wide_invariant(__read_implies(v, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__read_offset_bytes(v)) / (sizeof(double))) / (n)) % (n)) >= (2))) & ((n) >= (((((__read_offset_bytes(v)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((n) >= ((((__read_offset_bytes(v)) / (sizeof(double))) % (n)) + (2)))) & ((((__read_offset_bytes(v)) / (sizeof(double))) % (n)) >= (1))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(v)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(q, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1))) + (2)))) & (((((__read_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1))) >= (1))) & ((n) >= ((((__read_offset_bytes(q)) / (sizeof(double))) % (n)) + (2)))) & ((((__read_offset_bytes(q)) / (sizeof(double))) % (n)) >= (1))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1)))) % (1048576)) == (0))));
    }
}
