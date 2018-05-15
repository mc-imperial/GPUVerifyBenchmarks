//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel19(__global float *a3, __global float *a4, __global float *b1, __global float *b2, __global float *xp1, __global float *xp2, __global float *y2, __global float *yp1, __global float *yp2, int h, int w, long c0, long c1_0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((w) >= ((c0) + (1)))) & ((c0) >= (0))) & (((h) + (c1_0)) >= (1))) & ((c1_0) <= (0)));
      // shared
      y2[c0 * h + -c1_0] = ((((a3[0] * xp1[0]) + (a4[0] * xp2[0])) + (b1[0] * yp1[0])) + (b2[0] * yp2[0]));
      __function_wide_invariant(__write_implies(y2, ((c0) == ((((__write_offset_bytes(y2)) / (sizeof(float))) / (h)) % (w))) & (((((__write_offset_bytes(y2)) / (sizeof(float))) % (h)) + (c1_0)) == (0))));
    }
}
