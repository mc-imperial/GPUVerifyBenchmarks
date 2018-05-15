//pass
//--gridDim=64               --blockDim=256

template<class TData> __global__ void testKernel(TData *d_odata, TData *d_idata, int numElements);
template __global__ void testKernel<int>(int *d_odata, int *d_idata, int numElements);

template<class TData> __global__ void testKernel(
    TData *d_odata,
    TData *d_idata,
    int numElements
)
{
    const int        tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;

    for (int pos = tid; pos < numElements; pos += numThreads)
    {
        d_odata[pos] = d_idata[pos];
    }
}
