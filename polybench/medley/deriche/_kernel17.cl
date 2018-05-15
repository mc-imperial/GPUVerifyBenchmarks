//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel17(__global float *xp1, int h, int w, long c0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((((h) >= (0)) & ((h) <= (2147483647))) & ((w) <= (2147483647))) & ((w) >= ((c0) + (1)))) & ((c0) >= (0)));
      // shared
      xp1[0] = 0.F;
    }
}
