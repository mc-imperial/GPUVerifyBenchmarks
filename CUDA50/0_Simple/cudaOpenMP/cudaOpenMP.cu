//pass
//--gridDim=[64,1,1]       --blockDim=[128,1,1]

__global__ void kernelAddConstant(int *g_a, const int b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_a[idx] += b;
}
