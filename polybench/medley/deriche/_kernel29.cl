//PASS
//--local_size=[1] --num_groups=[1]

__kernel void kernel29(__global float *imgOut, __global float *tm1, int h, int w, long c0, long c1_0)
{
  __requires(h == 256);
  __requires(w == 1024);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires(((((((h) <= (2147483647)) & ((w) <= (2147483647))) & ((h) >= ((c0) + (1)))) & ((c0) >= (0))) & ((w) >= ((c1_0) + (1)))) & ((c1_0) >= (0)));
      // shared
      tm1[0] = imgOut[c1_0 * h + c0];
      __function_wide_invariant(__read_implies(imgOut, ((c0) == (((__read_offset_bytes(imgOut)) / (sizeof(float))) % (h))) & ((c1_0) == ((((__read_offset_bytes(imgOut)) / (sizeof(float))) / (h)) % (w)))));
    }
}
