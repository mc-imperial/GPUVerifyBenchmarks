//pass
//--local_size=[256] --num_groups=[12]

#include "../macros.h"

#define NC  4
#define COARSE_GENERAL
// #define COARSE_SPEC NC

__kernel void
ComputePhiMag_GPU(__global float* phiR, __global float* phiI, __global float* phiMag, int numK) {
  int indexK = get_global_id(0);
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}
