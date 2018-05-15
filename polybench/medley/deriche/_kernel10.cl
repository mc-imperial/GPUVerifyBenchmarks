//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel10(__global float *a3, __global float *a7, float alpha, __global float *k, int h, int w)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) >= (1)) & ((h) <= (2147483647))) & ((w) <= (0))) & ((w) >= (-2147483647))) | (((((h) >= (0)) & ((h) <= (2147483647))) & ((w) >= (1))) & ((w) <= (2147483647)))) | ((((h) == (0)) & ((w) <= (0))) & ((w) >= (-2147483648))));
      // shared
      a3[0] = (a7[0] = ((k[0] * exp((float) (-alpha))) * (alpha + 1.F)));
    }
}
