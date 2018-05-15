//pass
//--gridDim=[32768,1,1]    --blockDim=[512,1,1]

__global__ void init_array(int *g_data, int *factor, int num_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i<num_iterations; i++)
    {
        g_data[idx] += *factor;    // non-coalesced on purpose, to burn time
    }
}
