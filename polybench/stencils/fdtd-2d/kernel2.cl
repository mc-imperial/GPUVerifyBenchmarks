//PASS
//--local_size=[32] --num_groups=[16]

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel2(__global double *_fict_, __global double *ey, int ny, int tmax, int nx, long c0)
{
  __requires(ny == 512);
  __requires(tmax == 256);
  __requires(nx == 1024);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);
    __local double shared__fict_[1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((((((ny) >= (1)) & ((ny) <= (2147483647))) & ((tmax) <= (2147483647))) & ((nx) <= (2147483647))) & ((nx) >= (-2147483647))) & ((tmax) >= ((c0) + (1)))) & ((c0) >= (0)));
      if (t0 == 0)
        shared__fict_[0] = _fict_[c0];
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      for (long c1 = 32 * b0; c1 < ny; c1 += 1048576)
        if (ny >= t0 + c1 + 1) {
          // shared
          ey[0 * ny + (t0 + c1)] = shared__fict_[0];
        }
      __function_wide_invariant(__write_implies(ey, ((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__write_offset_bytes(ey)) / (sizeof(double))) / (ny)) % (__ite((nx) >= (2), nx, 1))) == (0))) & ((ny) >= ((((__write_offset_bytes(ey)) / (sizeof(double))) % (ny)) + (1)))) & ((((__write_offset_bytes(ey)) / (sizeof(double))) % (ny)) >= (0))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(ey)) / (sizeof(double))) % (ny))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(_fict_, ((t0) == (0)) & ((c0) == (((__read_offset_bytes(_fict_)) / (sizeof(double))) % (tmax)))));
    }
}
