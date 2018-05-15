//PASS
//--local_size=[32,16] --num_groups=[32,8]

__kernel void kernel24(__global float *c1, __global float *imgOut, __global float *y1, __global float *y2, int h, int w)
{
  __requires(h == 256);
  __requires(w == 1024);
    long b0 = get_group_id(0), b1_0 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);
    __local float shared_c1;

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires(((((h) >= (1)) & ((h) <= (2147483647))) & ((w) >= (1))) & ((w) <= (2147483647)));
      if (t0 == 0 && t1 == 0)
        shared_c1 = *c1;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      for (long c0 = 32 * b0; c0 < w; c0 += 8192)
        if (w >= t0 + c0 + 1)
          for (long c1_0 = 32 * b1_0; c1_0 < h; c1_0 += 8192) {
            // shared
            for (long c3 = t1; c3 <= min(31, h - c1_0 - 1); c3 += 16)
              imgOut[(t0 + c0) * h + (c1_0 + c3)] = (shared_c1 * (y1[(t0 + c0) * h + (c1_0 + c3)] + y2[(t0 + c0) * h + (c1_0 + c3)]));
          }
      __function_wide_invariant(__write_implies(imgOut, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((w) >= (((((__write_offset_bytes(imgOut)) / (sizeof(float))) / (h)) % (w)) + (1)))) & (((((__write_offset_bytes(imgOut)) / (sizeof(float))) / (h)) % (w)) >= (0))) & ((h) >= ((((__write_offset_bytes(imgOut)) / (sizeof(float))) % (h)) + (1)))) & ((((__write_offset_bytes(imgOut)) / (sizeof(float))) % (h)) >= (0))) & (((((__write_offset_bytes(imgOut)) / (sizeof(float))) % (h)) % (8192)) >= ((32) * (b1_0)))) & ((((32) * (b1_0)) + (31)) >= ((((__write_offset_bytes(imgOut)) / (sizeof(float))) % (h)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(imgOut)) / (sizeof(float))) / (h)) % (w))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(imgOut)) / (sizeof(float))) % (h))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(y2, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((w) >= (((((__read_offset_bytes(y2)) / (sizeof(float))) / (h)) % (w)) + (1)))) & (((((__read_offset_bytes(y2)) / (sizeof(float))) / (h)) % (w)) >= (0))) & ((h) >= ((((__read_offset_bytes(y2)) / (sizeof(float))) % (h)) + (1)))) & ((((__read_offset_bytes(y2)) / (sizeof(float))) % (h)) >= (0))) & (((((__read_offset_bytes(y2)) / (sizeof(float))) % (h)) % (8192)) >= ((32) * (b1_0)))) & ((((32) * (b1_0)) + (31)) >= ((((__read_offset_bytes(y2)) / (sizeof(float))) % (h)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(y2)) / (sizeof(float))) / (h)) % (w))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(y2)) / (sizeof(float))) % (h))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(y1, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((w) >= (((((__read_offset_bytes(y1)) / (sizeof(float))) / (h)) % (w)) + (1)))) & (((((__read_offset_bytes(y1)) / (sizeof(float))) / (h)) % (w)) >= (0))) & ((h) >= ((((__read_offset_bytes(y1)) / (sizeof(float))) % (h)) + (1)))) & ((((__read_offset_bytes(y1)) / (sizeof(float))) % (h)) >= (0))) & (((((__read_offset_bytes(y1)) / (sizeof(float))) % (h)) % (8192)) >= ((32) * (b1_0)))) & ((((32) * (b1_0)) + (31)) >= ((((__read_offset_bytes(y1)) / (sizeof(float))) % (h)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(y1)) / (sizeof(float))) / (h)) % (w))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(y1)) / (sizeof(float))) % (h))) % (16)) == (0))));
    }
}
