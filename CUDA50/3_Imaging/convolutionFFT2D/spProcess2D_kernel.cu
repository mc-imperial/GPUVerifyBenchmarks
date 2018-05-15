//pass
//--gridDim=4096             --blockDim=256

#include "common.h"

__global__ void spProcess2D_kernel(
    fComplex *d_Dst,
    fComplex *d_SrcA,
    fComplex *d_SrcB,
    uint DY,
    uint DX,
    uint threadCount,
    float phaseBase,
    float c
)
{
    __requires(DY == 2048);
    __requires(DX == 1024);
    __requires(threadCount == 1048576);

    const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId >= threadCount)
    {
        return;
    }

    uint x, y, i = threadId;
    udivmod(i, DX, x);
    udivmod(i, DY / 2, y);

    const uint offset = i * DY * DX;

    //Avoid overwrites in rows 0 and DY / 2 by different threads (left and right halves)
    //Otherwise correctness for in-place transformations is affected
    if ((y == 0) && (x > DX / 2))
    {
        return;
    }

    fComplex twiddle;

    //Process y = [0 .. DY / 2 - 1] U [DY - (DY / 2) + 1 .. DY - 1]
    {
        const uint pos1 = offset +          y * DX +          x;
        const uint pos2 = offset + mod(y, DY) * DX + mod(x, DX);

        fComplex D1 = LOAD_FCOMPLEX_A(pos1);
        fComplex D2 = LOAD_FCOMPLEX_A(pos2);
        fComplex K1 = LOAD_FCOMPLEX_B(pos1);
        fComplex K2 = LOAD_FCOMPLEX_B(pos2);
        getTwiddle(twiddle, phaseBase * (float)x);

        spPostprocessC2C(D1, D2, twiddle);
        spPostprocessC2C(K1, K2, twiddle);
        mulAndScale(D1, K1, c);
        mulAndScale(D2, K2, c);
        spPreprocessC2C(D1, D2, twiddle);

        d_Dst[pos1] = D1;
        d_Dst[pos2] = D2;
    }

    if (y == 0)
    {
        const uint pos1 = offset + (DY / 2) * DX +          x;
        const uint pos2 = offset + (DY / 2) * DX + mod(x, DX);

        fComplex D1 = LOAD_FCOMPLEX_A(pos1);
        fComplex D2 = LOAD_FCOMPLEX_A(pos2);
        fComplex K1 = LOAD_FCOMPLEX_B(pos1);
        fComplex K2 = LOAD_FCOMPLEX_B(pos2);

        spPostprocessC2C(D1, D2, twiddle);
        spPostprocessC2C(K1, K2, twiddle);
        mulAndScale(D1, K1, c);
        mulAndScale(D2, K2, c);
        spPreprocessC2C(D1, D2, twiddle);

        d_Dst[pos1] = D1;
        d_Dst[pos2] = D2;
    }
}
