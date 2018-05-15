//PASS
//--local_size=[32] --num_groups=[32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel19(__global double *a, __global double *b, __global double *d, __global double *f, __global double *p, __global double *q, __global double *u, int n, int tsteps, long c0)
{
  __requires(n == 1024);
  __requires(tsteps == 512);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_a;
    __local double shared_b;
    __local double shared_d;
    __local double shared_f;

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires((((((n) >= (3)) & ((n) <= (2147483647))) & ((tsteps) <= (2147483647))) & ((c0) >= (1))) & ((tsteps) >= (c0)));
      if (t0 == 0) {
        shared_a = *a;
        shared_b = *b;
        shared_d = *d;
        shared_f = *f;
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      for (long c1 = 32 * b0; c1 < n - 1; c1 += 1048576)
        for (long c2 = 0; c2 < n - 1; c2 += 32) {
          if (n >= t0 + c1 + 2 && t0 + c1 >= 1) {
            // shared
            for (long c4 = max(0, -c2 + 1); c4 <= min(31, n - c2 - 2); c4 += 1)
              q[(t0 + c1) * n + (c2 + c4)] = ((((((-shared_d) * u[(c2 + c4) * n + (t0 + c1 - 1)]) + ((1 + (2 * shared_d)) * u[(c2 + c4) * n + (t0 + c1)])) - (shared_f * u[(c2 + c4) * n + (t0 + c1 + 1)])) - (shared_a * q[(t0 + c1) * n + (c2 + c4 - 1)])) / ((shared_a * p[(t0 + c1) * n + (c2 + c4 - 1)]) + shared_b));
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
      __function_wide_invariant(__write_implies(q, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__write_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1))) + (2)))) & (((((__write_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1))) >= (1))) & ((n) >= ((((__write_offset_bytes(q)) / (sizeof(double))) % (n)) + (2)))) & ((((__write_offset_bytes(q)) / (sizeof(double))) % (n)) >= (1))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1)))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(q, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1))) + (2)))) & (((((__read_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1))) >= (1))) & ((n) >= ((((__read_offset_bytes(q)) / (sizeof(double))) % (n)) + (3)))) & ((((__read_offset_bytes(q)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(q)) / (sizeof(double))) / (n)) % ((n) - (1)))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(u, (((((((((t0) <= (31)) & ((t0) >= (-1048544))) & ((n) >= (((((__read_offset_bytes(u)) / (sizeof(double))) / (n)) % ((n) - (1))) + (2)))) & (((((__read_offset_bytes(u)) / (sizeof(double))) / (n)) % ((n) - (1))) >= (1))) & ((n) >= (((t0) + ((32) * (__ite(((((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30))<0), -((-(((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30))+(32)-1)/(32)), (((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30))/(32))))) + (2)))) & (((t0) + ((32) * (__ite(((((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30))<0), -((-(((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30))+(32)-1)/(32)), (((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30))/(32))))) >= (1))) & (((((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30)) % (32)) >= (30))) & (((b0) + ((32768) * (((((-((((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30)) % (32))) - ((32) * (b0))) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (1048574)) / (1048576)))) == ((((-(t0)) + (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (30)) / (32)))) | ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(u)) / (sizeof(double))) / (n)) % ((n) - (1))) + (2)))) & (((((__read_offset_bytes(u)) / (sizeof(double))) / (n)) % ((n) - (1))) >= (1))) & ((n) >= ((((__read_offset_bytes(u)) / (sizeof(double))) % (n)) + (3)))) & ((((__read_offset_bytes(u)) / (sizeof(double))) % (n)) >= (0))) & (((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(u)) / (sizeof(double))) % (n))) + (1048575)) % (1048576)) == (0)))));
      __function_wide_invariant(__read_implies(p, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1))) + (2)))) & (((((__read_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1))) >= (1))) & ((n) >= ((((__read_offset_bytes(p)) / (sizeof(double))) % (n)) + (3)))) & ((((__read_offset_bytes(p)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(p)) / (sizeof(double))) / (n)) % ((n) - (1)))) % (1048576)) == (0))));
    }
}
