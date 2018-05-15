//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *alpha, __global double *r, int n)
{
  __requires(n == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((n) <= (2147483647)) & ((n) >= (-2147483648)));
      // shared
      alpha[0] = (-r[0]);
      __function_wide_invariant(__read_implies(r, (((__read_offset_bytes(r)) / (sizeof(double))) % (__ite((n) >= (2), n, 1))) == (0)));
    }
}
