//pass
//--gridDim=[8192,1,1]     --blockDim=[512,1,1]

__global__ void incKernel(int *g_out, int *g_in, int N, int inner_reps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        for (int i=0; i<inner_reps; ++i)
        {
            g_out[idx] = g_in[idx] + 1;
        }
    }
}
