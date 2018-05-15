//pass
//--gridDim=[32768,1,1]    --blockDim=[512,1,1]

__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}
