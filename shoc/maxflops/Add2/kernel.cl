//pass
//--num_groups=16384 --local_size=128

#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void Add2(__global double *data, int nIters) {
  int gid = get_global_id(0), globalSize = get_global_size(0);
  double s = data[gid];
  double2 s0 = s + (double2)(0,0.1);
  for (int j=0 ; j<nIters ; ++j) {
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
 s0=10.f-s0;
  }
   data[gid] = s0.s0+s0.s1;
}
