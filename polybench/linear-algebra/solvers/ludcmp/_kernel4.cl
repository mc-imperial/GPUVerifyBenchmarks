//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel4(__global double *b, __global double *w, int n, long c0)
{
  __requires(n == 256);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((n) >= ((c0) + (1)))) & ((c0) >= (0)));
      // shared
      w[0] = b[c0];
      __function_wide_invariant(__read_implies(b, (c0) == (((__read_offset_bytes(b)) / (sizeof(double))) % (n))));
    }
}
