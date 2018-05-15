//pass
//--gridDim=512 --blockDim=384 --no-inline

#include "common.h"

__global__ void
SobelTex(Pixel *pSobelOriginal, unsigned int Pitch,
         int w, int h, float fScale, int blockOperation, pointFunction_t pPointOperation)
{
    __requires(Pitch == 512);
    __requires(w == 512);
    __requires(h == 512);
    __requires(blockOperation == 0 | blockOperation == 1);
    __requires(blockFunction_table[0] == ComputeSobel);
    __requires(blockFunction_table[1] == ComputeBox);
    __requires(pPointOperation == Threshold | pPointOperation == NULL);
    unsigned char *pSobel =
        (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);
    unsigned char tmp = 0;

    for (int i = threadIdx.x;
        __global_invariant(i % blockDim.x == threadIdx.x),
        __global_invariant(__write_implies(pSobelOriginal, __write_offset_bytes(pSobelOriginal)/sizeof(Pixel)%Pitch%blockDim.x == threadIdx.x)),
        i < w; i += blockDim.x)
    {
        unsigned char pix00 = tex2D(tex, (float) i-1, (float) blockIdx.x-1);
        unsigned char pix01 = tex2D(tex, (float) i+0, (float) blockIdx.x-1);
        unsigned char pix02 = tex2D(tex, (float) i+1, (float) blockIdx.x-1);
        unsigned char pix10 = tex2D(tex, (float) i-1, (float) blockIdx.x+0);
        unsigned char pix11 = tex2D(tex, (float) i+0, (float) blockIdx.x+0);
        unsigned char pix12 = tex2D(tex, (float) i+1, (float) blockIdx.x+0);
        unsigned char pix20 = tex2D(tex, (float) i-1, (float) blockIdx.x+1);
        unsigned char pix21 = tex2D(tex, (float) i+0, (float) blockIdx.x+1);
        unsigned char pix22 = tex2D(tex, (float) i+1, (float) blockIdx.x+1);
        tmp = (*(blockFunction_table[blockOperation]))(pix00, pix01, pix02,
                                                       pix10, pix11, pix12,
                                                       pix20, pix21, pix22, fScale);

        if (pPointOperation != NULL)
        {
            tmp = (*pPointOperation)(tmp, 150.0);
        }

        pSobel[i] = tmp;
    }
}
