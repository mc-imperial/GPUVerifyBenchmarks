//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel1(__global float *a1, __global float *a2, __global float *b1, __global float *b2, __global float *imgIn, __global float *xm1, __global float *y1, __global float *ym1, __global float *ym2, int h, int w, long c0, long c1_0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((c0) >= (1))) & ((w) >= (c0))) & ((h) >= ((c1_0) + (1)))) & ((c1_0) >= (0)));
      // shared
      y1[(c0 - 1) * h + c1_0] = ((((a1[0] * imgIn[(c0 - 1) * h + c1_0]) + (a2[0] * xm1[0])) + (b1[0] * ym1[0])) + (b2[0] * ym2[0]));
      __function_wide_invariant(__write_implies(y1, ((c0) == (((((__write_offset_bytes(y1)) / (sizeof(float))) / (h)) % (w)) + (1))) & ((c1_0) == (((__write_offset_bytes(y1)) / (sizeof(float))) % (h)))));
      __function_wide_invariant(__read_implies(imgIn, ((c0) == (((((__read_offset_bytes(imgIn)) / (sizeof(float))) / (h)) % (w)) + (1))) & ((c1_0) == (((__read_offset_bytes(imgIn)) / (sizeof(float))) % (h)))));
    }
}
