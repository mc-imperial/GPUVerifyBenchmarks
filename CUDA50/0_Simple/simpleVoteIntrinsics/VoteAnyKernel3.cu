//pass
//--gridDim=1                --blockDim=32

#ifndef IMPLEMENT_VOTE_INTRINSICS
__device__ unsigned int all(unsigned int);
__device__ unsigned int any(unsigned int);
#endif

__global__ void VoteAnyKernel3(bool *info, int warp_size)
{
    int tx = threadIdx.x;
    bool *offs = info + (tx * 3);

    // The following should hold true for the second and third warp
    *offs = any((tx >= (warp_size * 3) / 2));
    // The following should hold true for the "upper half" of the second warp,
    // and all of the third warp
    *(offs + 1) = (tx >= (warp_size * 3) / 2? true: false);

    // The following should hold true for the third warp only
    if (all((tx >= (warp_size * 3) / 2)))
    {
        *(offs + 2) = true;
    }
}
