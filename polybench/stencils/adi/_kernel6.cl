//PASS
//--local_size=[1] --num_groups=[1]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel6(__global double *B1, int n, int tsteps)
{
  __requires(n == 1024);
  __requires(tsteps == 512);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((n) >= (0)) & ((n) <= (2147483647))) & ((tsteps) <= (2147483647))) & ((tsteps) >= (-2147483648)));
      // shared
      B1[0] = 2.;
    }
}
