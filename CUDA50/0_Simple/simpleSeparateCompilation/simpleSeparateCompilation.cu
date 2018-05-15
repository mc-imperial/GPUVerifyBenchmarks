//pass
//--gridDim=[1,1,1]        --blockDim=[1024,1,1] --no-inline

__device__ float multiplyByTwo(float number)
{
    return number * 2.0f;
}

__device__ float divideByTwo(float number)
{
    return number * 0.5f;
}

typedef unsigned int uint;
typedef float(*deviceFunc)(float);

__global__ void transformVector(float *v, deviceFunc f, uint size)
{
    __requires(f == multiplyByTwo | f == divideByTwo);
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        v[tid] = (*f)(v[tid]);
    }
}
