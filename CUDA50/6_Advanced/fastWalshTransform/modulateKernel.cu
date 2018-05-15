//pass
//--gridDim=128              --blockDim=256

__global__ void modulateKernel(float *d_A, float *d_B, int N)
{
    int        tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    float     rcpN = 1.0f / (float)N;

    for (int pos = tid; pos < N; pos += numThreads)
    {
        d_A[pos] *= d_B[pos] * rcpN;
    }
}
