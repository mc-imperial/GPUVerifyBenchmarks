//pass
//--gridDim=[1,1,1]        --blockDim=[128,1,1]

#ifndef IMPLEMENT_VOTE_INTRINSICS
__device__ unsigned int any(unsigned int);
#endif

__global__ void VoteAnyKernel1(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;

    result[tx] = any(input[tx]);
}

