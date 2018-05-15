//PASS
//--local_size=[32] --num_groups=[128]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *A, double alpha, __global double *w, __global double *x, int n)
{
  __requires(n == 4096);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_A[32][32];
    double private_w[1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((n) >= (1)) & ((n) <= (2147483647)));
      for (long c0 = 32 * b0; c0 < n; c0 += 1048576) {
        if (n >= t0 + c0 + 1)
          private_w[0] = w[t0 + c0];
        for (long c1 = 0; c1 < n; c1 += 32) {
          if (n >= t0 + c1 + 1)
            for (long c2 = 0; c2 <= min(31, n - c0 - 1); c2 += 1)
              shared_A[c2][t0] = A[(c0 + c2) * n + (t0 + c1)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (n >= t0 + c0 + 1) {
            // shared
            for (long c3 = 0; c3 <= min(31, n - c1 - 1); c3 += 1)
              private_w[0] = (private_w[0] + ((alpha * shared_A[t0][c3]) * x[c1 + c3]));
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (n >= t0 + c0 + 1)
          w[t0 + c0] = private_w[0];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
      __function_wide_invariant(__write_implies(w, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__write_offset_bytes(w)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(w)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(w)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(w, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__read_offset_bytes(w)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(w)) / (sizeof(double))) % (n)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(w)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(x, (((n) >= ((((32) * (b0)) + (t0)) + (1))) & ((n) >= ((((__read_offset_bytes(x)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(x)) / (sizeof(double))) % (n)) >= (0))));
      __function_wide_invariant(__read_implies(A, ((((((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & ((((((-32) * (b0)) + ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) + (1048544)) % (1048576)) >= (1048544))) & ((((t0) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) % (32)) == (0))));
    }
}
