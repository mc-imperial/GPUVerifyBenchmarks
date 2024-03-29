//PASS
//--local_size=[32] --num_groups=[2]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *A, int n, long c0)
{
  __requires(n == 64);
  __requires(c0 < 3 * n - 2); // Possible overflow in __requires
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires((((n) <= (2147483647)) & (((3) * (n)) >= ((c0) + (4)))) & (((2) * (__ite(((((2) * (c0)) + (2))<0), -((-(((2) * (c0)) + (2))+(3)-1)/(3)), (((2) * (c0)) + (2))/(3)))) >= ((c0) + (2))));
      for (long c1 = 32 * b0 + 1048576 * ((-96 * b0 + c0 + 3145635) / 3145728); c1 <= min(n - 1, c0 / 2); c1 += 1048576)
        if (n >= t0 + c1 + 1 && c0 >= 2 * t0 + 2 * c1 && 3 * t0 + 3 * c1 >= c0 + 1) {
          // shared
          A[(t0 + c1) * n + (t0 + c1)] -= (A[(t0 + c1) * n + (-2 * t0 + c0 - 2 * c1)] * A[(t0 + c1) * n + (-2 * t0 + c0 - 2 * c1)]);
        }
      __function_wide_invariant(__write_implies(A, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & (((3) * ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) >= ((c0) + (1)))) & ((c0) >= ((2) * ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & ((c0) == (((2) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (1048576)) == (0))) | ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)) + (1)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n)))) & (((3) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) >= ((c0) + (1)))) & ((c0) >= ((2) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) % (1048576)) == (0)))));
    }
}
