//PASS
//--local_size=[32,16] --num_groups=[32,8]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel3(__global double *A, __global double *Q, __global double *R, int n, int m, long c0)
{
  __requires(n == 1024);
  __requires(m == 256);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires(((((((n) <= (2147483647)) & ((m) >= (1))) & ((m) <= (2147483647))) & ((c0) >= (5))) & (((5) * (n)) >= ((c0) + (5)))) & (((5) * ((((2) * (c0)) + (2)) / (3))) >= (((3) * (c0)) + (3))));
      if ((3 * n >= c0 + 1 && 5 * ((2 * c0 + 2) / 3) >= 3 * c0 + 5) || (c0 >= 3 * n && 5 * ((n + c0 + 1) / 2) >= 3 * c0 + 5))
        for (long c1 = max(32 * b0 + 8192 * floord(-n - 64 * b0 + c0 - 64, 16384) + 8192, 32 * b0 + 8192 * ((-96 * b0 + c0 + 24480) / 24576)); c1 < 2 * c0 / 5; c1 += 8192)
          if (n + 2 * t0 + 2 * c1 + 1 >= c0 && 3 * t0 + 3 * c1 + 2 >= c0 && 2 * c0 >= 5 * t0 + 5 * c1 + 5)
            for (long c2 = 32 * b1; c2 < m; c2 += 8192) {
              // shared
              for (long c4 = t1; c4 <= min(31, m - c2 - 1); c4 += 16)
                A[(c2 + c4) * n + (-2 * t0 + c0 - 2 * c1 - 2)] = (A[(c2 + c4) * n + (-2 * t0 + c0 - 2 * c1 - 2)] - (Q[(c2 + c4) * n + (3 * t0 - c0 + 3 * c1 + 2)] * R[(3 * t0 - c0 + 3 * c1 + 2) * n + (-2 * t0 + c0 - 2 * c1 - 2)]));
            }
      __function_wide_invariant(__write_implies(A, (((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((m) >= (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((n) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((c0) >= (((3) * (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) + (2)))) & (((5) * (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) >= (c0))) & ((((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) % (8192)))) & ((((t1) - ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (m))) % (16)) == (0))) & ((((((((64) * (b0)) + ((2) * (t0))) + (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) - (c0)) + (2)) % (16384)) == (0))));
      __function_wide_invariant(__read_implies(Q, (((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((m) >= (((((__read_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__read_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((((__read_offset_bytes(Q)) / (sizeof(double))) % (n)) >= (0))) & ((c0) >= (((5) * (((__read_offset_bytes(Q)) / (sizeof(double))) % (n))) + (5)))) & ((((3) * (n)) + ((2) * (((__read_offset_bytes(Q)) / (sizeof(double))) % (n)))) >= ((c0) + (1)))) & ((((((__read_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= (((((__read_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m)) % (8192)))) & ((((t1) - ((((__read_offset_bytes(Q)) / (sizeof(double))) / (n)) % (m))) % (16)) == (0))) & ((((((((96) * (b0)) + ((3) * (t0))) + ((24575) * (((__read_offset_bytes(Q)) / (sizeof(double))) % (n)))) - (c0)) + (2)) % (24576)) == (0))));
      __function_wide_invariant(__read_implies(R, (((((((((32) * (b1)) >= ((t1) + (1))) & ((m) >= ((((32) * (b1)) + (t1)) + (1)))) & (((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((((__read_offset_bytes(R)) / (sizeof(double))) % (n)) >= (((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((n) >= ((((__read_offset_bytes(R)) / (sizeof(double))) % (n)) + (1)))) & ((c0) == ((((2) * ((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n))) + ((3) * (((__read_offset_bytes(R)) / (sizeof(double))) % (n)))) + (2)))) & (((((((32) * (b0)) + (t0)) + ((8191) * ((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)))) - (((__read_offset_bytes(R)) / (sizeof(double))) % (n))) % (8192)) == (0))) | ((((((((b1) == (0)) & ((m) >= ((t1) + (1)))) & (((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((((__read_offset_bytes(R)) / (sizeof(double))) % (n)) >= (((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((n) >= ((((__read_offset_bytes(R)) / (sizeof(double))) % (n)) + (1)))) & ((c0) == ((((2) * ((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n))) + ((3) * (((__read_offset_bytes(R)) / (sizeof(double))) % (n)))) + (2)))) & (((((((32) * (b0)) + (t0)) + ((8191) * ((((__read_offset_bytes(R)) / (sizeof(double))) / (n)) % (n)))) - (((__read_offset_bytes(R)) / (sizeof(double))) % (n))) % (8192)) == (0)))));
      __function_wide_invariant(__read_implies(A, (((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((m) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) >= (0))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((c0) >= (((3) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (2)))) & (((5) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) >= (c0))) & ((((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m)) % (8192)))) & ((((t1) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (m))) % (16)) == (0))) & ((((((((64) * (b0)) + ((2) * (t0))) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) - (c0)) + (2)) % (16384)) == (0))));
    }
}
