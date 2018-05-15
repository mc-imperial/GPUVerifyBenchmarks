//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel3(__global double *A, int n, long c0)
{
  __requires(n == 64);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((n) <= (2147483647)) & (((3) * (n)) >= ((c0) + (3)))) & ((c0) >= (0))) & (((c0) % (3)) == (0)));
      // shared
      A[(c0 / 3) * n + (c0 / 3)] = sqrt(A[(c0 / 3) * n + (c0 / 3)]);
      __function_wide_invariant(__write_implies(A, ((((__write_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) & ((c0) == ((3) * ((((__write_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))))));
      __function_wide_invariant(__read_implies(A, ((((__read_offset_bytes(A)) / (sizeof(double))) % (n)) == ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))) & ((c0) == ((3) * ((((__read_offset_bytes(A)) / (sizeof(double))) / (n)) % (n))))));
    }
}
