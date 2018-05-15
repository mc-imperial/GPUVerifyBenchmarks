//pass
//--blockDim=1024 --gridDim=128

#include <cuda.h>

//--------------------------------------------------------------------------------------
// File: ComputeEngine.h
//
// This is an AMPC++ implementation of a compute shader. It transforms a shape with a
// rotation of an angle THETA. 
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------


#define THETA 3.1415f/1024  

__global__ void run(float* data_refY, float* data_refX)
{
  // Rotate the vertex by angle THETA
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  data_refY[idx] = data_refY[idx] * cos(THETA) - data_refX[idx] * sin(THETA);
  data_refX[idx] = data_refY[idx] * sin(THETA) + data_refX[idx] * cos(THETA);
#ifdef MUTATION
  data_refX[idx+1] = data_refX[idx+1];
   /* BUGINJECT: ADD_ACCESS, UP */
#endif
}


