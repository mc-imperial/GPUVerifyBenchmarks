//PASS
//--local_size=[32] --num_groups=[2]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel5(__global double *A, int n, long c0)
{
  __requires(n == 64);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (1))) & (((3) * (__ite(((((n) + (c0)) + (1))<0), -((-(((n) + (c0)) + (1))+(2)-1)/(2)), (((n) + (c0)) + (1))/(2)))) >= (((2) * (c0)) + (4))));
      for (long c1 = max(32 * b0, 32 * b0 + 1048576 * floord(-n - 64 * b0 + c0 - 62, 2097152) + 1048576); c1 <= (c0 - 1) / 3; c1 += 1048576)
        if (n + 2 * t0 + 2 * c1 >= c0 + 1 && c0 >= 3 * t0 + 3 * c1 + 1) {
          // shared
          A[(-2 * t0 + c0 - 2 * c1) * n + (t0 + c1)] /= A[(t0 + c1) * n + (t0 + c1)];
        }
      __function_wide_invariant(__write_implies(A, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & ((c0) == (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + ((2) * (((__write_offset_bytes(A)) / (sizeof(double))) % (n)))))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(A)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= (0))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & ((c0) >= (((3) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) + (1)))) & (((n) + ((2) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) >= ((c0) + (1)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (1048576)) == (0))) | ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & ((c0) == (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + ((2) * (((__read_offset_bytes(A)) / (sizeof(double))) % (n)))))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) % (1048576)) == (0)))));
    }
}
