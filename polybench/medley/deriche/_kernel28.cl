//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel28(__global float *a5, __global float *a6, __global float *b1, __global float *b2, __global float *imgOut, __global float *tm1, __global float *y1, __global float *ym1, __global float *ym2, int h, int w, long c0, long c1_0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((h) >= ((c0) + (1)))) & ((c0) >= (0))) & ((w) >= ((c1_0) + (1)))) & ((c1_0) >= (0)));
      // shared
      y1[c1_0 * h + c0] = ((((a5[0] * imgOut[c1_0 * h + c0]) + (a6[0] * tm1[0])) + (b1[0] * ym1[0])) + (b2[0] * ym2[0]));
      __function_wide_invariant(__write_implies(y1, ((c0) == (((__write_offset_bytes(y1)) / (sizeof(float))) % (h))) & ((c1_0) == ((((__write_offset_bytes(y1)) / (sizeof(float))) / (h)) % (w)))));
      __function_wide_invariant(__read_implies(imgOut, ((c0) == (((__read_offset_bytes(imgOut)) / (sizeof(float))) % (h))) & ((c1_0) == ((((__read_offset_bytes(imgOut)) / (sizeof(float))) / (h)) % (w)))));
    }
}
