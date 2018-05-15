//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel0(__global double *temp2, int n, int m, long c0, long c1)
{
  __requires(n == 1024);
  __requires(m == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((n) <= (2147483647)) & ((m) <= (2147483647))) & ((m) >= ((c0) + (1)))) & ((c0) >= (0))) & ((n) >= ((c1) + (1)))) & ((c1) >= (0)));
      // shared
      temp2[0] = 0;
    }
}
