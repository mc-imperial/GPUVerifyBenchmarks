//pass
//--blockDim=[17,17] --gridDim=[1,1]

#include <cuda.h>

// code example	for blog: Use extent instead of grid class - Sample 2
//                 created by: Tamer Afify              Date:1/1/2012

//This sample shows how to replace grid with extent in the 
//previously illustrated image blur solution.
//For code porting process follow those three simple steps;
//1. Wherever grid type or array/aray_view value is used replace with extent
//2. If array is constructed with a grid origin index value, then whenever 
//   this array is used add the origin index to its index value.
//3. If the compute domain grid - for parallel_for_each – is constructed with origin, 
//   add this origin to every index use in the kernel.

// Note: to compile this code you need to use Visual Studio 2011 Beta Release

#define width 17
#define height 17


__global__ void boxblur(float* blurimage, float* img, int originX, int originY)
{
    int idxX = blockIdx.x*blockDim.x + threadIdx.x;
    int idxY = blockIdx.y*blockDim.y + threadIdx.y;
    float r = 0.0f;
    int samples = 0;
    idxX += originX;
    idxY += originY;

    for (int dy = -1; dy <= 1; dy++)
    {
      for (int dx = -1; dx <= 1; dx++)
      {
        r += img[(idxY+dy)*width + idxX + dx];
        samples++;
      }
    }

    blurimage[idxY*width + idxX] = r/samples;
#if MUTATION
    blurimage[idxY*width + idxX + 1] = blurimage[idxY*width + idxX + 1];
#endif

}
