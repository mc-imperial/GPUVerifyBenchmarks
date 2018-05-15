//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel4(__global double *nrm, int n, int m, long c0)
{
  __requires(n == 1024);
  __requires(m == 256);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((n) <= (2147483647)) & ((m) <= (2147483647))) & ((m) >= (-2147483648))) & (((5) * (n)) >= ((c0) + (2)))) & ((c0) >= (0))) & (((c0) % (5)) == (0)));
      // shared
      nrm[0] = 0;
    }
}
