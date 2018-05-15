//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel4(__global float *y1, __global float *ym1, int h, int w, long c0, long c1_0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((c0) >= (1))) & ((w) >= (c0))) & ((h) >= ((c1_0) + (1)))) & ((c1_0) >= (0)));
      // shared
      ym1[0] = y1[(c0 - 1) * h + c1_0];
      __function_wide_invariant(__read_implies(y1, ((c0) == (((((__read_offset_bytes(y1)) / (sizeof(float))) / (h)) % (w)) + (1))) & ((c1_0) == (((__read_offset_bytes(y1)) / (sizeof(float))) % (h)))));
    }
}
