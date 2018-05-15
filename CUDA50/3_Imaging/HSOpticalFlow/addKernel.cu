//pass
//--gridDim=[1200,1,1]     --blockDim=[256,1,1]
  
__global__
void AddKernel(const float *op1, const float *op2, int count, float *sum)
{
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos >= count) return;

    sum[pos] = op1[pos] + op2[pos];
}
