//pass
//--gridDim=4096             --blockDim=256

#include "common.h"

__global__ void spPreprocess2D_kernel(
    fComplex *d_Dst,
    fComplex *d_Src,
    uint DY,
    uint DX,
    uint threadCount,
    uint padding,
    float phaseBase
)
{
    __requires(DY == 2048);
    __requires(DX == 1024);
    __requires(threadCount == 1048576);
    __requires(padding == 16);

    const uint threadId = blockIdx.x *  blockDim.x + threadIdx.x;

    if (threadId >= threadCount)
    {
        return;
    }

    uint x, y, i = threadId;
    udivmod(i, DX / 2, x);
    udivmod(i, DY, y);

    //Avoid overwrites in columns 0 and DX / 2 by different threads (lower and upper halves)
    if ((x == 0) && (y > DY / 2))
    {
        return;
    }

    const uint srcOffset = i * DY * (DX + padding);
    const uint dstOffset = i * DY * DX;

    //Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
    {
        const uint  loadPos1 = srcOffset +          y * (DX + padding) +        x;
        const uint  loadPos2 = srcOffset + mod(y, DY) * (DX + padding) + (DX - x);
        const uint storePos1 = dstOffset +          y * DX +          x;
        const uint storePos2 = dstOffset + mod(y, DY) * DX + mod(x, DX);

        fComplex D1 = LOAD_FCOMPLEX(loadPos1);
        fComplex D2 = LOAD_FCOMPLEX(loadPos2);

        fComplex twiddle;
        getTwiddle(twiddle, phaseBase * (float)x);
        spPreprocessC2C(D1, D2, twiddle);

        d_Dst[storePos1] = D1;
        d_Dst[storePos2] = D2;
    }

    //Process x = DX / 2
    if (x == 0)
    {
        const uint  loadPos1 = srcOffset +          y * (DX + padding) + DX / 2;
        const uint  loadPos2 = srcOffset + mod(y, DY) * (DX + padding) + DX / 2;
        const uint storePos1 = dstOffset +          y * DX + DX / 2;
        const uint storePos2 = dstOffset + mod(y, DY) * DX + DX / 2;

        fComplex D1 = LOAD_FCOMPLEX(loadPos1);
        fComplex D2 = LOAD_FCOMPLEX(loadPos2);

        //twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(-dir * j * PI / 2)
        fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
        spPreprocessC2C(D1, D2, twiddle);

        d_Dst[storePos1] = D1;
        d_Dst[storePos2] = D2;
    }
}
