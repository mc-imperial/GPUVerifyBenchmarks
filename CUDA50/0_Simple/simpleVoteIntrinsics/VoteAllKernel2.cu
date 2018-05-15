//pass
//--gridDim=[1,1,1]        --blockDim=[128,1,1]

#ifndef IMPLEMENT_VOTE_INTRINSICS
__device__ unsigned int all(unsigned int);
#endif

__global__ void VoteAllKernel2(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;

    result[tx] = all(input[tx]);
}
