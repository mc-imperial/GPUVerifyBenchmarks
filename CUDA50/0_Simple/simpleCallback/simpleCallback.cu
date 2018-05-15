//pass
//--gridDim=[196,1,1]      --blockDim=[512,1,1]
  
__global__
void incKernel(int *data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        data[i]++;
}
