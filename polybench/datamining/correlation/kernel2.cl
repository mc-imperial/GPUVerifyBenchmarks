//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *data, double float_n, __global double *mean, __global double *stddev, int m, int n)
{
  __requires(m == 512);
  __requires(n == 1024);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared_data[32][32];
    double private_mean[1];
    double private_stddev[1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((m) >= (1)) & ((m) <= (2147483647))) & ((n) <= (2147483647))) & ((n) >= (-2147483648)));
      for (long c0 = 32 * b0; c0 < m; c0 += 1048576) {
        if (m >= t0 + c0 + 1) {
          private_mean[0] = mean[t0 + c0];
          if (n >= 1)
            private_stddev[0] = stddev[t0 + c0];
        }
        for (long c1 = 0; c1 < n; c1 += 32) {
          if (m >= t0 + c0 + 1)
            for (long c2 = 0; c2 <= min(31, n - c1 - 1); c2 += 1)
              shared_data[c2][t0] = data[(c1 + c2) * m + (t0 + c0)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (m >= t0 + c0 + 1) {
            // shared
            {
              if (c1 == 0)
                private_mean[0] /= float_n;
              for (long c3 = 0; c3 <= min(31, n - c1 - 1); c3 += 1) {
                private_stddev[0] += ((shared_data[c3][t0] - private_mean[0]) * (shared_data[c3][t0] - private_mean[0]));
                shared_data[c3][t0] -= private_mean[0];
              }
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (m >= t0 + c0 + 1)
            for (long c2 = 0; c2 <= min(31, n - c1 - 1); c2 += 1)
              data[(c1 + c2) * m + (t0 + c0)] = shared_data[c2][t0];
        }
        if (n <= 0) {
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (m >= t0 + c0 + 1) {
            // shared
            private_mean[0] /= float_n;
          }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        } else if (m >= t0 + c0 + 1)
          stddev[t0 + c0] = private_stddev[0];
        if (m >= t0 + c0 + 1)
          mean[t0 + c0] = private_mean[0];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
      __function_wide_invariant(__write_implies(data, (((((n) >= (((((__write_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1))) & (((((__write_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__write_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & ((((__write_offset_bytes(data)) / (sizeof(double))) % (m)) >= (t0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(data)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
      __function_wide_invariant(__write_implies(mean, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= ((((__write_offset_bytes(mean)) / (sizeof(double))) % (m)) + (1)))) & ((((__write_offset_bytes(mean)) / (sizeof(double))) % (m)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(mean)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
      __function_wide_invariant(__write_implies(stddev, ((((((n) >= (1)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= ((((__write_offset_bytes(stddev)) / (sizeof(double))) % (m)) + (1)))) & ((((__write_offset_bytes(stddev)) / (sizeof(double))) % (m)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(stddev)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(data, (((((n) >= (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) + (1))) & (((((__read_offset_bytes(data)) / (sizeof(double))) / (m)) % (n)) >= (0))) & ((m) >= ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(data)) / (sizeof(double))) % (m)) >= (t0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(data)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(mean, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= ((((__read_offset_bytes(mean)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(mean)) / (sizeof(double))) % (m)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(mean)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(stddev, ((((((n) >= (1)) & ((((32) * (b0)) + (t0)) >= (0))) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= ((((__read_offset_bytes(stddev)) / (sizeof(double))) % (m)) + (1)))) & ((((__read_offset_bytes(stddev)) / (sizeof(double))) % (m)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(stddev)) / (sizeof(double))) % (m))) % (1048576)) == (0))));
    }
}
