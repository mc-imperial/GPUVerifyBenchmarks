//pass
//--gridDim=512 --blockDim=384

#include "common.h"

#define min(x,y) (x < y ? x : y)
#define max(x,y) (x < y ? y : x)

__global__ void
SobelCopyImage(Pixel *pSobelOriginal, unsigned int Pitch,
               int w, int h, float fscale)
{
    __requires(w == 512);
    __requires(Pitch == 512);

    unsigned char *pSobel =
        (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x;
         __global_invariant(i % blockDim.x == threadIdx.x),
         __global_invariant(__write_implies(pSobelOriginal, __write_offset_bytes(pSobelOriginal)%Pitch%blockDim.x == threadIdx.x)),
         i < w; i += blockDim.x)
    {
        pSobel[i] = min(max((tex2D(tex, (float) i, (float) blockIdx.x) * fscale), 0.f), 255.f);
    }
}
