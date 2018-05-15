//pass
//--blockDim=[17,17] --gridDim=[1,1]

#include <cuda.h>

// code example	for blog: Use extent instead of grid class - Sample 1
//                 created by: Tamer Afify              Date:1/1/2012

//This is a sample function for using the grid class to do an image blur
//The pixel blur can be performed by changing every pixel color RBG band to 
//the arithmetic average of this pixel value with all its 8 neighbors’ pixels.

//The grid offset feature can be of great benefit when the compute domain origin 
//is different from the data origin. In other words, (0,0) for the data is not 
//matching the (0,0) starting point for computation.

//In this sample we will use this feature to blur the inner image box without the 
//boarder pixels as they don’t have 8 neighbors pixel. So the compute domain origin 
//is (1, 1) in the data index. And also compute domain extent is smaller than data 
//extent by 2 rows and 2 columns.

// Note: to compile this code you need to use C++ AMP Developer Preview destributed
// During the TAP progrm.

#define width 17
#define height 17

__global__ void boxblur(float* blurimage, float* img)
{
    int idxX = blockIdx.x*blockDim.x + threadIdx.x;
    int idxY = blockIdx.y*blockDim.y + threadIdx.y;

    float r = 0.0f;
    int samples = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
      for (int dx = -1; dx <= 1; dx++)
      {
        r += img[(idxY + dy)*width + idxX + dx];
        samples++;
      }
    }

    blurimage[idxY*width + idxX] = r/samples;
#if MUTATION
    blurimage[idxY*width + idxX + 1] = blurimage[idxY*width + idxX + 1];
#endif

}
