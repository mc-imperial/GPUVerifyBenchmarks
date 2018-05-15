//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel6(float alpha, __global float *b1, int h, int w)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) >= (1)) & ((h) <= (2147483647))) & ((w) <= (0))) & ((w) >= (-2147483647))) | (((((h) >= (0)) & ((h) <= (2147483647))) & ((w) >= (1))) & ((w) <= (2147483647)))) | ((((h) == (0)) & ((w) <= (0))) & ((w) >= (-2147483648))));
      // shared
      b1[0] = pow((float) 2.F, -alpha);
    }
}
