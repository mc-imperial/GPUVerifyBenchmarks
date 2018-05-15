//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel9(__global double *corr, int m, int n)
{
  __requires(m == 512);
  __requires(n == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((m) >= (1)) & ((m) <= (2147483647))) & ((n) <= (2147483647))) & ((n) >= (-2147483648)));
      // shared
      corr[(m - 1) * (m >= 2 ? m : 1) + (m - 1)] = 1.;
      __function_wide_invariant(__write_implies(corr, ((((((__write_offset_bytes(corr)) / (sizeof(double))) / (__ite((m) >= (2), m, 1))) % (__ite((m) >= (2), m, 1))) + (1)) == (m)) & (((((__write_offset_bytes(corr)) / (sizeof(double))) % (__ite((m) >= (2), m, 1))) + (1)) == (m))));
    }
}
