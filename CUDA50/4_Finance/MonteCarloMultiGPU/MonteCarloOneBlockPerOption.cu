//pass
//--gridDim=256 --blockDim=256 --warp-sync=32 -DUNROLL_REDUCTION

// Notes:
// There are two reductions given in reduction.h
// If the unrolled reduction is used then we need to rely on implicit warpsyncs

//Loop unrolled
//--gridDim=256 --blockDim=256 --warp-sync=32 -DUNROLL_REDUCTION
//Nested loops
//--gridDim=256 --blockDim=256

//REQUIRES: const array as formal (imperial edit)

#ifndef DOUBLE_PRECISION
typedef float real;
#else
typedef double real;
#endif

#include "reduction.h"
#define MAX_OPTIONS 512
#define THREAD_N 256

typedef struct
{
    real S;
    real X;
    real MuByT;
    real VBySqrtT;
} __TOptionData;

typedef struct {
    real Expected;
    real Confidence;
} __TOptionValue;

#if 0 // imperial edit
static __device__ __constant__ __TOptionData d_OptionData[MAX_OPTIONS];
static __device__ __TOptionValue d_CallValue[MAX_OPTIONS];
#endif

__device__ static __attribute__((always_inline)) float endCallValue(float S, float X, float r, float MuByT, float VBySqrtT)
{
    float callValue = S * __expf(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}

__global__ void MonteCarloOneBlockPerOption(
    __TOptionData *d_OptionData,   // imperial edit
    __TOptionValue * d_CallValue,  // imperial edit
    curandState *rngStates,
    int pathN)
{
    const int SUM_N = THREAD_N;
    __shared__ real s_SumCall[SUM_N];
    __shared__ real s_Sum2Call[SUM_N];

    const int optionIndex = blockIdx.x;
    const real        S = d_OptionData[optionIndex].S;
    const real        X = d_OptionData[optionIndex].X;
    const real    MuByT = d_OptionData[optionIndex].MuByT;
    const real VBySqrtT = d_OptionData[optionIndex].VBySqrtT;

    // determine global thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy random number state to local memory for efficiency
    curandState localState = rngStates[tid];

    //Cycle through the entire samples array:
    //derive end stock price for each path
    //accumulate partial integrals into intermediate shared memory buffer
    for (int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x)
    {
        __TOptionValue sumCall = {0, 0};

        for (int i = iSum; i < pathN; i += SUM_N)
        {
            real              r = curand_normal(&localState);
            real      callValue = endCallValue(S, X, r, MuByT, VBySqrtT);
            sumCall.Expected   += callValue;
            sumCall.Confidence += callValue * callValue;
        }

        s_SumCall[iSum]  = sumCall.Expected;
        s_Sum2Call[iSum] = sumCall.Confidence;
    }

    // store random number state back to global memory
    rngStates[tid] = localState;

    //Reduce shared memory accumulators
    //and write final result to global memory
    sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call);

    if (threadIdx.x == 0)
    {
        __TOptionValue t = {s_SumCall[0], s_Sum2Call[0]};
        d_CallValue[optionIndex] = t;
    }
}
