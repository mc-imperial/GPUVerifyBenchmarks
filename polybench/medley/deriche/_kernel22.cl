//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel22(__global float *imgIn, __global float *xp1, int h, int w, long c0, long c1_0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((w) >= ((c0) + (1)))) & ((c0) >= (0))) & (((h) + (c1_0)) >= (1))) & ((c1_0) <= (0)));
      // shared
      xp1[0] = imgIn[c0 * h + -c1_0];
      __function_wide_invariant(__read_implies(imgIn, ((c0) == ((((__read_offset_bytes(imgIn)) / (sizeof(float))) / (h)) % (w))) & (((((__read_offset_bytes(imgIn)) / (sizeof(float))) % (h)) + (c1_0)) == (0))));
    }
}
