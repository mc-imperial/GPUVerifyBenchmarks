//pass
//--gridDim=[32,32,1] --blockDim=[16,16,1]

#include "common.h"
  
__global__ void
d_render_regular(uint *d_output, uint imageW, uint imageH,
                 float density, float brightness,
                 float transferOffset, float transferScale, float transferWeight = 0.0f)
{
    __requires(imageW == 32*16 /*gridDim.x*blockDim.x*/);

    d_render<TF_SINGLE_1D>(d_output,imageW,imageH,density,brightness,transferOffset,transferScale,transferWeight);
}
