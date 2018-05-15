//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel23(__global float *y2, __global float *yp1, int h, int w, long c0, long c1_0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((w) >= ((c0) + (1)))) & ((c0) >= (0))) & (((h) + (c1_0)) >= (1))) & ((c1_0) <= (0)));
      // shared
      yp1[0] = y2[c0 * h + -c1_0];
      __function_wide_invariant(__read_implies(y2, ((c0) == ((((__read_offset_bytes(y2)) / (sizeof(float))) / (h)) % (w))) & (((((__read_offset_bytes(y2)) / (sizeof(float))) % (h)) + (c1_0)) == (0))));
    }
}
