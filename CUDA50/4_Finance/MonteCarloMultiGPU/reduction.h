template<class T, unsigned int blockSize>
__device__ static __attribute__((always_inline)) void sumReduceSharedMem(volatile T *sum, volatile T *sum2, int tid)
{
    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sum[tid] += sum[tid + 256];
            sum2[tid] += sum2[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sum[tid] += sum[tid + 128];
            sum2[tid] += sum2[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sum[tid] += sum[tid +  64];
            sum2[tid] += sum2[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        if (blockSize >=  64)
        {
            sum[tid] += sum[tid + 32];
            sum2[tid] += sum2[tid + 32];
        }

        if (blockSize >=  32)
        {
            sum[tid] += sum[tid + 16];
            sum2[tid] += sum2[tid + 16];
        }

        if (blockSize >=  16)
        {
            sum[tid] += sum[tid +  8];
            sum2[tid] += sum2[tid +  8];
        }

        if (blockSize >=   8)
        {
            sum[tid] += sum[tid +  4];
            sum2[tid] += sum2[tid +  4];
        }

        if (blockSize >=   4)
        {
            sum[tid] += sum[tid +  2];
            sum2[tid] += sum2[tid +  2];
        }

        if (blockSize >=   2)
        {
            sum[tid] += sum[tid +  1];
            sum2[tid] += sum2[tid +  1];
        }
    }
}

template<class T, int SUM_N, int blockSize>
__device__ static __attribute__((always_inline)) void sumReduce(T *sum, T *sum2)
{
#ifdef UNROLL_REDUCTION

    for (int pos = threadIdx.x; pos < SUM_N; pos += blockSize)
    {
        __syncthreads();
        sumReduceSharedMem<T, blockSize>(sum, sum2, pos);
    }

#else

    for (int stride = SUM_N / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        for (int pos = threadIdx.x;
             __global_invariant(__implies(stride <= threadIdx.x, !__write(sum))),
             __global_invariant(__read_implies(sum, (__read_offset_bytes(sum)/sizeof(T) % stride % blockSize == threadIdx.x))),
             __global_invariant(__implies(stride <= threadIdx.x, !__write(sum2))),
             __global_invariant(__read_implies(sum2, (__read_offset_bytes(sum2)/sizeof(T) % stride % blockSize == threadIdx.x))),
             pos < stride; pos += blockSize)
        {
            sum[pos] += sum[pos + stride];
            sum2[pos] += sum2[pos + stride];
        }
    }

#endif
}
