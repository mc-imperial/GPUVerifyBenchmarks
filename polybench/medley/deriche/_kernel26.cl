//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel26(__global float *ym2, int h, int w, long c0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((w) >= (-2147483647))) & ((h) >= ((c0) + (1)))) & ((c0) >= (0)));
      // shared
      ym2[0] = 0.F;
    }
}
