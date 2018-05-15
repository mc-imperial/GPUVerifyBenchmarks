//pass
//--gridDim=[4,1,1]        --blockDim=[512,1,1]

__global__ void
initValue(float *od, float value)
{
    // position of write into global memory
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    od[index] = value;

    // sync after each decomposition step
    __syncthreads();
}
