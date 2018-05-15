//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *L, __global double *x, int n, long c0)
{
  __requires(n == 512);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_x_0[32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (2))) & (((2) * (n)) >= ((c0) + (2))));
      for (long c1 = max(32 * b0, 32 * b0 + 1048576 * floord(-n - 32 * b0 + c0 - 32, 1048576) + 1048576);
           __global_invariant(__read_implies(shared_x_0, __read_offset_bytes(shared_x_0)/sizeof(double) == t0)), c1 < c0 / 2; c1 += 1048576) {
        if (t0 + c0 >= c1 + 32 && n + c1 + 31 >= t0 + c0)
          shared_x_0[t0] = x[t0 + c0 - c1 - 32];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (n + t0 + c1 >= c0 && c0 >= 2 * t0 + 2 * c1 + 2) {
          // shared
          shared_x_0[-t0 + 31] -= (L[(-t0 + c0 - c1 - 1) * n + (t0 + c1)] * x[t0 + c1]);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (n + c1 + 31 >= t0 + c0 && 2 * t0 + c0 >= 2 * c1 + 64)
          x[t0 + c0 - c1 - 32] = shared_x_0[t0];
      }
      __function_wide_invariant(__write_implies(x, ((((n) >= ((((__write_offset_bytes(x)) / (sizeof(double))) % (n)) + (1))) & ((c0) >= ((((__write_offset_bytes(x)) / (sizeof(double))) % (n)) + (1)))) & (((2) * (((__write_offset_bytes(x)) / (sizeof(double))) % (n))) >= (c0))) & ((((((((32) * (b0)) + ((1048575) * (t0))) + (((__write_offset_bytes(x)) / (sizeof(double))) % (n))) - (c0)) + (32)) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(x, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((((__read_offset_bytes(x)) / (sizeof(double))) % (n)) >= (0))) & ((c0) >= (((2) * (((__read_offset_bytes(x)) / (sizeof(double))) % (n))) + (2)))) & (((n) + (((__read_offset_bytes(x)) / (sizeof(double))) % (n))) >= (c0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(x)) / (sizeof(double))) % (n))) % (1048576)) == (0))) | ((((((n) >= ((((__read_offset_bytes(x)) / (sizeof(double))) % (n)) + (1))) & ((((__read_offset_bytes(x)) / (sizeof(double))) % (n)) >= (0))) & (((t0) + (c0)) >= ((((__read_offset_bytes(x)) / (sizeof(double))) % (n)) + (32)))) & ((((2) * (((__read_offset_bytes(x)) / (sizeof(double))) % (n))) + (62)) >= (((2) * (t0)) + (c0)))) & ((((((((32) * (b0)) + ((1048575) * (t0))) + (((__read_offset_bytes(x)) / (sizeof(double))) % (n))) - (c0)) + (32)) % (1048576)) == (0)))));
      __function_wide_invariant(__read_implies(L, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(L)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(L)) / (sizeof(double))) / (n)) % (n)) >= ((((__read_offset_bytes(L)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(L)) / (sizeof(double))) % (n)) >= (0))) & ((c0) == ((((((__read_offset_bytes(L)) / (sizeof(double))) / (n)) % (n)) + (((__read_offset_bytes(L)) / (sizeof(double))) % (n))) + (1)))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(L)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
    }
}
