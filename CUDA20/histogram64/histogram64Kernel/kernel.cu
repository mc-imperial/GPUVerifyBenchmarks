//pass
//--blockDim=[128,1] --gridDim=[64,1]

#include <cuda.h>

#define BIN_COUNT 64

////////////////////////////////////////////////////////////////////////////////
// GPU-specific definitions
////////////////////////////////////////////////////////////////////////////////
//Fast mul on G8x / G9x / G100
#define IMUL(a, b) a * b

//Threads block size for histogram64Kernel()
//Preferred to be a multiple of 64 (refer to the supplied whitepaper)
//REVISIT: 192 is not a pow2 so is very slow to prove
//#define THREAD_N 192
#define THREAD_N 128


////////////////////////////////////////////////////////////////////////////////
// If threadPos == threadIdx.x, there are always  4-way bank conflicts,
// since each group of 16 threads (half-warp) accesses different bytes,
// but only within 4 shared memory banks. Having shuffled bits of threadIdx.x 
// as in histogram64GPU(), each half-warp accesses different shared memory banks
// avoiding any bank conflicts at all.
// Refer to the supplied whitepaper for detailed explanations.
////////////////////////////////////////////////////////////////////////////////
// REVISIT: this inline syntax does not work
__device__ inline void addData64(unsigned char *s_Hist, int threadPos, unsigned int data) __attribute__((always_inline));
__device__ inline void addData64(unsigned char *s_Hist, int threadPos, unsigned int data) {
    s_Hist[threadPos + IMUL(data, THREAD_N)]++;
}

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute gridDim.x partial histograms
////////////////////////////////////////////////////////////////////////////////
__global__ void histogram64Kernel(unsigned int *d_Result, unsigned int *d_Data, int dataN){
    //Encode thread index in order to avoid bank conflicts in s_Hist[] access:
    //each half-warp accesses consecutive shared memory banks
    //and the same bytes within the banks

    const int threadPos =
        //[31 : 6] <== [31 : 6]
        ((threadIdx.x & (~63)) >> 0) |
        //[5  : 2] <== [3  : 0]
        ((threadIdx.x &    15) << 2) |
        //[1  : 0] <== [5  : 4]
        ((threadIdx.x &    48) >> 4);

    //Per-thread histogram storage
    __shared__ unsigned char s_Hist[THREAD_N * BIN_COUNT];

    //Flush shared memory
    for(int i = 0;
             i < BIN_COUNT / 4; i++) {
      //         ((unsigned int *)s_Hist)[threadIdx.x + i * THREAD_N] = 0; 
      s_Hist[threadIdx.x + i * THREAD_N] = 0;
    }

    __syncthreads();

    ////////////////////////////////////////////////////////////////////////////
    // Cycle through current block, update per-thread histograms
    // Since only 64-bit histogram of 8-bit input data array is calculated,
    // only highest 6 bits of each 8-bit data element are extracted,
    // leaving out 2 lower bits.
    ////////////////////////////////////////////////////////////////////////////
    unsigned int data4;
    for(int pos = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
        // These two invariants are strong (fast to prove) but only valid if THREAD_N/blockDim.x is a pow2
        __global_invariant(
          __implies(__is_pow2(THREAD_N),
            __read_implies(s_Hist,
              __mod_pow2( __read_offset_bytes(s_Hist) - (((data4 >> 26) & 63) * THREAD_N), THREAD_N) == threadPos))),
        __global_invariant(
          __implies(__is_pow2(THREAD_N),
            __write_implies(s_Hist,
              __mod_pow2(__write_offset_bytes(s_Hist) - (((data4 >> 26) & 63) * THREAD_N), THREAD_N) == threadPos))),
                  pos < dataN; pos += IMUL(blockDim.x, gridDim.x)){
        data4 = d_Data[pos];
        addData64(s_Hist, threadPos, (data4 >>  2) & 0x3FU);
        addData64(s_Hist, threadPos, (data4 >> 10) & 0x3FU);
        addData64(s_Hist, threadPos, (data4 >> 18) & 0x3FU);
        addData64(s_Hist, threadPos, (data4 >> 26) & 0x3FU);
    }

    __syncthreads();

    ////////////////////////////////////////////////////////////////////////////
    // Merge per-thread histograms into per-block and write to global memory.
    // Start accumulation positions for half-warp each thread are shifted
    // in order to avoid bank conflicts. 
    // See supplied whitepaper for detailed explanations.
    ////////////////////////////////////////////////////////////////////////////
     /* BUGINJECT: ADD_BARRIER, DOWN */
    if(threadIdx.x < BIN_COUNT){
#ifdef MUTATION
        __syncthreads();
#endif

        unsigned int sum = 0;
        const int value = threadIdx.x;

        const int valueBase = IMUL(value, THREAD_N);
        const int  startPos = (threadIdx.x & 15) * 4;

        //Threads with non-zero start positions wrap around the THREAD_N border
        // REVISIT: loop index clash with loop0 rewritten to use 'j' instead
        for(int j = 0, accumPos = startPos; j < THREAD_N; j++){
            sum += s_Hist[valueBase + accumPos];
            accumPos++;
            if(accumPos == THREAD_N) accumPos = 0;
        }

        d_Result[blockIdx.x * BIN_COUNT + value] = sum;
    }
}
