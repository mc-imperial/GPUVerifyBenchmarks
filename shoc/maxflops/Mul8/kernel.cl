//pass
//--num_groups=16384 --local_size=128

#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void Mul8(__global double *data, int nIters) {
  int gid = get_global_id(0), globalSize = get_global_size(0);
  double s = data[gid]-data[gid]+0.999f;
  double8 s0 = s + (double8)(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7);
  for (int j=0 ; j<nIters ; ++j) {
 s0=s0*s0*1.01f;
 s0=s0*s0*1.01f;
 s0=s0*s0*1.01f;
 s0=s0*s0*1.01f;
 s0=s0*s0*1.01f;
 s0=s0*s0*1.01f;
 s0=s0*s0*1.01f;
 s0=s0*s0*1.01f;
 s0=s0*s0*1.01f;
  }
   data[gid] = s0.s0+s0.s1+s0.s2+s0.s3+s0.s4+s0.s5+s0.s6+s0.s7;
}
