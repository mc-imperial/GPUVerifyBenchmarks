//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *L, __global double *x, int n, long c0)
{
  __requires(n == 512);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((n) <= (2147483647)) & ((c0) >= (1))) & (((2) * (n)) >= ((c0) + (1)))) & ((((c0) - (1)) % (2)) == (0)));
      // shared
      x[(c0 - 1) / 2] = (x[(c0 - 1) / 2] / L[((c0 - 1) / 2) * n + ((c0 - 1) / 2)]);
      __function_wide_invariant(__write_implies(x, (c0) == (((2) * (((__write_offset_bytes(x)) / (sizeof(double))) % (n))) + (1))));
      __function_wide_invariant(__read_implies(L, ((((__read_offset_bytes(L)) / (sizeof(double))) % (n)) == ((((__read_offset_bytes(L)) / (sizeof(double))) / (n)) % (n))) & ((c0) == (((2) * ((((__read_offset_bytes(L)) / (sizeof(double))) / (n)) % (n))) + (1)))));
      __function_wide_invariant(__read_implies(x, (c0) == (((2) * (((__read_offset_bytes(x)) / (sizeof(double))) % (n))) + (1))));
    }
}
