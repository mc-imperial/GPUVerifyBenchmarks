//PASS
//--local_size=[32] --num_groups=[32]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *A, __global double *B, int tsteps, int n, long c0)
{
  __requires(n == 1024);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((((tsteps) <= (2147483647)) & ((n) >= (3))) & ((n) <= (2147483647))) & ((tsteps) >= ((c0) + (1)))) & ((c0) >= (0)));
      for (long c1 = 32 * b0; c1 < n - 1; c1 += 1048576)
        if (n >= t0 + c1 + 2 && t0 + c1 >= 1) {
          // shared
          B[t0 + c1] = (0.33333 * ((A[t0 + c1 - 1] + A[t0 + c1]) + A[t0 + c1 + 1]));
        }
      __function_wide_invariant(__write_implies(B, (((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__write_offset_bytes(B)) / (sizeof(double))) % (n)) + (2)))) & ((((__write_offset_bytes(B)) / (sizeof(double))) % (n)) >= (1))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(B)) / (sizeof(double))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(A, (((((((t0) <= (31)) & ((t0) >= (-1048544))) & ((n) >= (((t0) + ((32) * (__ite(((((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30))<0), -((-(((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30))+(32)-1)/(32)), (((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30))/(32))))) + (2)))) & (((t0) + ((32) * (__ite(((((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30))<0), -((-(((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30))+(32)-1)/(32)), (((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30))/(32))))) >= (1))) & (((((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30)) % (32)) >= (30))) & (((b0) + ((32768) * (((((-((((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30)) % (32))) - ((32) * (b0))) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (1048574)) / (1048576)))) == ((((-(t0)) + (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (30)) / (32)))) | ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) + (3)))) & ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) >= (0))) & (((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(A)) / (sizeof(double))) % (n))) + (1048575)) % (1048576)) == (0)))));
    }
}
