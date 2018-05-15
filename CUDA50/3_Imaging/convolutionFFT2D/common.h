typedef unsigned int uint;
typedef float2 fComplex;

#define  USE_TEXTURE 1
#define POWER_OF_TWO 1

#if(USE_TEXTURE)
texture<float, 1, cudaReadModeElementType> texFloat;
#define   LOAD_FLOAT(i) tex1Dfetch(texFloat, i)
#define  SET_FLOAT_BASE checkCudaErrors( cudaBindTexture(0, texFloat, d_Src) )
#else
#define  LOAD_FLOAT(i) d_Src[i]
#define SET_FLOAT_BASE
#endif

__device__ static __attribute__((always_inline)) void mulAndScale(fComplex &a, const fComplex &b, const float &c)
{
    fComplex t = {c *(a.x * b.x - a.y * b.y), c *(a.y * b.x + a.x * b.y)};
    a = t;
}

#if(USE_TEXTURE)
texture<fComplex, 1, cudaReadModeElementType> texComplexA;
texture<fComplex, 1, cudaReadModeElementType> texComplexB;
#define    LOAD_FCOMPLEX(i) tex1Dfetch(texComplexA, i)
#define  LOAD_FCOMPLEX_A(i) tex1Dfetch(texComplexA, i)
#define  LOAD_FCOMPLEX_B(i) tex1Dfetch(texComplexB, i)

#define   SET_FCOMPLEX_BASE checkCudaErrors( cudaBindTexture(0, texComplexA,  d_Src) )
#define SET_FCOMPLEX_BASE_A checkCudaErrors( cudaBindTexture(0, texComplexA, d_SrcA) )
#define SET_FCOMPLEX_BASE_B checkCudaErrors( cudaBindTexture(0, texComplexB, d_SrcB) )
#else
#define    LOAD_FCOMPLEX(i)  d_Src[i]
#define  LOAD_FCOMPLEX_A(i) d_SrcA[i]
#define  LOAD_FCOMPLEX_B(i) d_SrcB[i]

#define   SET_FCOMPLEX_BASE
#define SET_FCOMPLEX_BASE_A
#define SET_FCOMPLEX_BASE_B
#endif

__device__ static __attribute__((always_inline)) void spPostprocessC2C(fComplex &D1, fComplex &D2, const fComplex &twiddle)
{
    float A1 = 0.5f * (D1.x + D2.x);
    float B1 = 0.5f * (D1.y - D2.y);
    float A2 = 0.5f * (D1.y + D2.y);
    float B2 = 0.5f * (D1.x - D2.x);

    D1.x = A1 + (A2 * twiddle.x + B2 * twiddle.y);
    D1.y = (A2 * twiddle.y - B2 * twiddle.x) + B1;
    D2.x = A1 - (A2 * twiddle.x + B2 * twiddle.y);
    D2.y = (A2 * twiddle.y - B2 * twiddle.x) - B1;
}

__device__ static __attribute__((always_inline)) void spPreprocessC2C(fComplex &D1, fComplex &D2, const fComplex &twiddle)
{
    float A1 = /* 0.5f * */ (D1.x + D2.x);
    float B1 = /* 0.5f * */ (D1.y - D2.y);
    float A2 = /* 0.5f * */ (D1.y + D2.y);
    float B2 = /* 0.5f * */ (D1.x - D2.x);

    D1.x = A1 - (A2 * twiddle.x - B2 * twiddle.y);
    D1.y = (B2 * twiddle.x + A2 * twiddle.y) + B1;
    D2.x = A1 + (A2 * twiddle.x - B2 * twiddle.y);
    D2.y = (B2 * twiddle.x + A2 * twiddle.y) - B1;
}

__device__ static __attribute__((always_inline)) void getTwiddle(fComplex &twiddle, float phase)
{
    __sincosf(phase, &twiddle.y, &twiddle.x);
}

__device__ static __attribute__((always_inline)) uint mod(uint a, uint DA)
{
    //(DA - a) % DA, assuming a <= DA
    return a ? (DA - a) : a;
}

__device__ static __attribute__((always_inline)) void udivmod(uint &dividend, uint divisor, uint &rem)
{
#if(!POWER_OF_TWO)
    rem = dividend % divisor;
    dividend /= divisor;
#else
    rem = dividend & (divisor - 1);
    dividend >>= (__ffs(divisor) - 1);
#endif
}
