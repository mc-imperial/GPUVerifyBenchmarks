//PASS
//--local_size=[32,16] --num_groups=[32,16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel1(__global double *ey, __global double *hz, int ny, int tmax, int nx, long c0)
{
  __requires(ny == 512);
  __requires(tmax == 256);
  __requires(nx == 1024);
    long b0 = get_group_id(0), b1 = get_group_id(1);
    long t0 = get_local_id(0), t1 = get_local_id(1);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    {
      __requires((((((((ny) >= (1)) & ((ny) <= (2147483647))) & ((tmax) <= (2147483647))) & ((nx) >= (2))) & ((nx) <= (2147483647))) & ((tmax) >= ((c0) + (1)))) & ((c0) >= (0)));
      for (long c1 = 32 * b0; c1 < nx; c1 += 8192)
        if (t0 + c1 >= 1 && nx >= t0 + c1 + 1)
          for (long c2 = 32 * b1; c2 < ny; c2 += 8192) {
            // shared
            for (long c4 = t1; c4 <= min(31, ny - c2 - 1); c4 += 16)
              ey[(t0 + c1) * ny + (c2 + c4)] = (ey[(t0 + c1) * ny + (c2 + c4)] - (0.5 * (hz[(t0 + c1) * ny + (c2 + c4)] - hz[(t0 + c1 - 1) * ny + (c2 + c4)])));
          }
      __function_wide_invariant(__write_implies(ey, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__write_offset_bytes(ey)) / (sizeof(double))) / (ny)) % (__ite((nx) >= (2), nx, 1))) >= (1))) & ((nx) >= (((((__write_offset_bytes(ey)) / (sizeof(double))) / (ny)) % (__ite((nx) >= (2), nx, 1))) + (1)))) & ((ny) >= ((((__write_offset_bytes(ey)) / (sizeof(double))) % (ny)) + (1)))) & ((((__write_offset_bytes(ey)) / (sizeof(double))) % (ny)) >= (0))) & (((((__write_offset_bytes(ey)) / (sizeof(double))) % (ny)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__write_offset_bytes(ey)) / (sizeof(double))) % (ny)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__write_offset_bytes(ey)) / (sizeof(double))) / (ny)) % (__ite((nx) >= (2), nx, 1)))) % (8192)) == (0))) & ((((t1) - (((__write_offset_bytes(ey)) / (sizeof(double))) % (ny))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(ey, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__read_offset_bytes(ey)) / (sizeof(double))) / (ny)) % (__ite((nx) >= (2), nx, 1))) >= (1))) & ((nx) >= (((((__read_offset_bytes(ey)) / (sizeof(double))) / (ny)) % (__ite((nx) >= (2), nx, 1))) + (1)))) & ((ny) >= ((((__read_offset_bytes(ey)) / (sizeof(double))) % (ny)) + (1)))) & ((((__read_offset_bytes(ey)) / (sizeof(double))) % (ny)) >= (0))) & (((((__read_offset_bytes(ey)) / (sizeof(double))) % (ny)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(ey)) / (sizeof(double))) % (ny)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(ey)) / (sizeof(double))) / (ny)) % (__ite((nx) >= (2), nx, 1)))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(ey)) / (sizeof(double))) % (ny))) % (16)) == (0))));
      __function_wide_invariant(__read_implies(hz, (((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & (((((__read_offset_bytes(hz)) / (sizeof(double))) / (ny)) % (nx)) >= (1))) & ((nx) >= (((((__read_offset_bytes(hz)) / (sizeof(double))) / (ny)) % (nx)) + (1)))) & ((ny) >= ((((__read_offset_bytes(hz)) / (sizeof(double))) % (ny)) + (1)))) & ((((__read_offset_bytes(hz)) / (sizeof(double))) % (ny)) >= (0))) & (((((__read_offset_bytes(hz)) / (sizeof(double))) % (ny)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(hz)) / (sizeof(double))) % (ny)) % (8192)))) & ((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(hz)) / (sizeof(double))) / (ny)) % (nx))) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(hz)) / (sizeof(double))) % (ny))) % (16)) == (0))) | (((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (8191))) & ((nx) >= (((((__read_offset_bytes(hz)) / (sizeof(double))) / (ny)) % (nx)) + (2)))) & (((((__read_offset_bytes(hz)) / (sizeof(double))) / (ny)) % (nx)) >= (0))) & ((ny) >= ((((__read_offset_bytes(hz)) / (sizeof(double))) % (ny)) + (1)))) & ((((__read_offset_bytes(hz)) / (sizeof(double))) % (ny)) >= (0))) & (((((__read_offset_bytes(hz)) / (sizeof(double))) % (ny)) % (8192)) >= ((32) * (b1)))) & ((((32) * (b1)) + (31)) >= ((((__read_offset_bytes(hz)) / (sizeof(double))) % (ny)) % (8192)))) & (((((((32) * (b0)) + (t0)) - ((((__read_offset_bytes(hz)) / (sizeof(double))) / (ny)) % (nx))) + (8191)) % (8192)) == (0))) & ((((t1) - (((__read_offset_bytes(hz)) / (sizeof(double))) % (ny))) % (16)) == (0)))));
    }
}
