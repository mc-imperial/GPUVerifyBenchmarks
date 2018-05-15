//PASS
//--local_size=[32] --num_groups=[32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *A, __global double *Q, __global double *R, int n, int m, long c0)
{
  __requires(n == 1024);
  __requires(m == 256);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    double private_R[1][1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires(((((((n) <= (2147483647)) & ((m) >= (1))) & ((m) <= (2147483647))) & (((5) * (n)) >= ((c0) + (6)))) & ((c0) >= (4))) & (((5) * ((((2) * (c0)) + (1)) / (3))) >= (((3) * (c0)) + (1))));
      for (long c1 = max(32 * b0 + 1048576 * floord(-n - 64 * b0 + c0 - 63, 2097152) + 1048576, 32 * b0 + 1048576 * ((-96 * b0 + c0 + 3145633) / 3145728)); c1 < (2 * c0 + 2) / 5; c1 += 1048576) {
        if (n + 2 * t0 + 2 * c1 >= c0 && 3 * t0 + 3 * c1 + 1 >= c0 && 2 * c0 >= 5 * t0 + 5 * c1 + 3) {
          private_R[0][0] = R[(3 * t0 - c0 + 3 * c1 + 1) * n + (-2 * t0 + c0 - 2 * c1 - 1)];
          for (long c2 = 0; c2 < m; c2 += 32) {
            // shared
            for (long c4 = 0; c4 <= min(31, m - c2 - 1); c4 += 1)
              private_R[0][0] += (Q[(c2 + c4) * n + (3 * t0 - c0 + 3 * c1 + 1)] * A[(c2 + c4) * n + (-2 * t0 + c0 - 2 * c1 - 1)]);
          }
          R[(3 * t0 - c0 + 3 * c1 + 1) * n + (-2 * t0 + c0 - 2 * c1 - 1)] = private_R[0][0];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
      __function_wide_invariant(__write_implies(R, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__write_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((((__write_offset_bytes(R)) / (sizeof(double))) % (n)) >= (((((__write_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((n) >= ((((__write_offset_bytes(R)) / (sizeof(double))) % (n)) + (1)))) & ((c0) == ((((2) * ((((__write_offset_bytes(R)) / (sizeof(double))) / (n)) % (n))) + ((3) * (((__write_offset_bytes(R)) / (sizeof(double))) % (n)))) + (1)))) & (((((((32) * (b0)) + (t0)) + ((1048575) * ((((__write_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)))) - (((__write_offset_bytes(R)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((c0) >= (((3) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (1)))) & (((5) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) >= ((c0) + (1)))) & ((((((((64) * (b0)) + ((2) * (t0))) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) - (c0)) + (1)) % (2097152)) == (0))));
      __function_wide_invariant(__read_implies(Q, ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((m) >= (((((__read_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__read_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((((__read_offset_bytes(Q)) / (sizeof(double))) % (n)) >= (0))) & ((c0) >= (((5) * (((__read_offset_bytes(Q)) / (sizeof(double))) % (n))) + (4)))) & ((((3) * (n)) + ((2) * (((__read_offset_bytes(Q)) / (sizeof(double))) % (n)))) >= ((c0) + (2)))) & ((((((((96) * (b0)) + ((3) * (t0))) + ((3145727) * (((__read_offset_bytes(Q)) / (sizeof(double))) % (n)))) - (c0)) + (1)) % (3145728)) == (0))));
      __function_wide_invariant(__read_implies(R, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((((__read_offset_bytes(R)) / (sizeof(double))) % (n)) >= (((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((n) >= ((((__read_offset_bytes(R)) / (sizeof(double))) % (n)) + (1)))) & ((c0) == ((((2) * ((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n))) + ((3) * (((__read_offset_bytes(R)) / (sizeof(double))) % (n)))) + (1)))) & (((((((32) * (b0)) + (t0)) + ((1048575) * ((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)))) - (((__read_offset_bytes(R)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
    }
}
