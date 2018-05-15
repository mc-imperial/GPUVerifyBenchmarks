//pass
//--blockDim=[64,1] --gridDim=[64,1]

#include <cuda.h>

#define BIN_COUNT 64

////////////////////////////////////////////////////////////////////////////////
// GPU-specific definitions
////////////////////////////////////////////////////////////////////////////////
//Fast mul on G8x / G9x / G100
#define IMUL(a, b) a * b

////////////////////////////////////////////////////////////////////////////////
// Merge blockN histograms into gridDim.x histograms
// blockDim.x == BIN_COUNT
// gridDim.x  == BLOCK_N2
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADS 64


__global__ void mergeHistogram64Kernel(
    unsigned int *d_Histogram,
    unsigned int *d_PartialHistograms,
    unsigned int blockN
){
    __shared__ unsigned int data[MERGE_THREADS];

    unsigned int sum = 0;
    for(unsigned int i = threadIdx.x; i < blockN; i += MERGE_THREADS) {
        sum += d_PartialHistograms[blockIdx.x + i * BIN_COUNT];
    }
    data[threadIdx.x] = sum;

    for(unsigned int stride = MERGE_THREADS / 2;
        stride > 0; stride >>= 1){
        __syncthreads();
         /* BUGINJECT: ADD_BARRIER, DOWN */
        if(threadIdx.x < stride) {
#ifdef MUTATION
            __syncthreads();
#endif
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if(threadIdx.x == 0)
        d_Histogram[blockIdx.x] = data[0];
}
