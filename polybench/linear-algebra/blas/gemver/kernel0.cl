//PASS
//--local_size=[32,16] --num_groups=[128,128]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, __global double *u1, __global double *u2, __global double *v1, __global double *v2, int n)
{
  __requires(n == 4096);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local double shared_u1[32];
    __local double shared_u2[32];
    __local double shared_v1[32];
    __local double shared_v2[32];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((n) >= (1)) & ((n) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 8192) {
        if (t0 == 0) {
          for (long c1 = t1; c1 <= min(31, n - c0 - 1); c1 += 16)
            shared_u1[c1] = u1[c0 + c1];
          for (long c1 = t1; c1 <= min(31, n - c0 - 1); c1 += 16)
            shared_u2[c1] = u2[c0 + c1];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        for (long c1 = 32 * b1; c1 < n; c1 += 8192) {
          if (t0 == 0) {
            for (long c2 = t1; c2 <= min(31, n - c1 - 1); c2 += 16)
              shared_v1[c2] = v1[c1 + c2];
            for (long c2 = t1; c2 <= min(31, n - c1 - 1); c2 += 16)
              shared_v2[c2] = v2[c1 + c2];
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (n >= t0 + c0 + 1) {
            // shared
            for (long c3 = t1; c3 <= min(31, n - c1 - 1); c3 += 16)
              A[(t0 + c0) * n + (c1 + c3)] = ((A[(t0 + c0) * n + (c1 + c3)] + (shared_u1[t0] * shared_v1[c3])) + (shared_u2[t0] * shared_v2[c3]));
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
      __function_wide_invariant(__write_implies(A, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((n) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & (((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(u2, (((((t0) == (0)) & ((n) >= ((((__read_offset_bytes(u2)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(u2)) / (sizeof(double))) % (n)) >= (0))) & (((((((32) * (b0)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(u2)) / (sizeof(double))) % (n)))) + (31)) % (8192)) <= (31))) & ((((t1) - (((__read_offset_bytes(u2)) / (sizeof(double))) % (n))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & (((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(v2, (((((t0) == (0)) & ((n) >= ((((__read_offset_bytes(v2)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(v2)) / (sizeof(double))) % (n)) >= (0))) & (((((((32) * (b1)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(v2)) / (sizeof(double))) % (n)))) + (31)) % (8192)) <= (31))) & ((((t1) - (((__read_offset_bytes(v2)) / (sizeof(double))) % (n))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(v1, (((((t0) == (0)) & ((n) >= ((((__read_offset_bytes(v1)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(v1)) / (sizeof(double))) % (n)) >= (0))) & (((((((32) * (b1)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(v1)) / (sizeof(double))) % (n)))) + (31)) % (8192)) <= (31))) & ((((t1) - (((__read_offset_bytes(v1)) / (sizeof(double))) % (n))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(u1, (((((t0) == (0)) & ((n) >= ((((__read_offset_bytes(u1)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(u1)) / (sizeof(double))) % (n)) >= (0))) & (((((((32) * (b0)) + ((7680) * (t1))) + ((511) * (((__read_offset_bytes(u1)) / (sizeof(double))) % (n)))) + (31)) % (8192)) <= (31))) & ((((t1) - (((__read_offset_bytes(u1)) / (sizeof(double))) % (n))) % (16)) == (0))));
    }
}
