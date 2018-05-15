//pass
//--gridDim=8 --blockDim=512

__global__ void simpleKernel(int *dst, int *src, int num)
{
    // Dummy kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] / num;
}
