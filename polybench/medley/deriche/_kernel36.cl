//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel36(__global float *a7, __global float *a8, __global float *b1, __global float *b2, __global float *tp1, __global float *tp2, __global float *y2, __global float *yp1, __global float *yp2, int h, int w, long c0, long c1_0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((h) >= ((c0) + (1)))) & ((c0) >= (0))) & (((w) + (c1_0)) >= (1))) & ((c1_0) <= (0)));
      // shared
      y2[-c1_0 * h + c0] = ((((a7[0] * tp1[0]) + (a8[0] * tp2[0])) + (b1[0] * yp1[0])) + (b2[0] * yp2[0]));
      __function_wide_invariant(__write_implies(y2, ((c0) == (((__write_offset_bytes(y2)) / (sizeof(float))) % (h))) & ((((((__write_offset_bytes(y2)) / (sizeof(float))) / (h)) % (w)) + (c1_0)) == (0))));
    }
}
