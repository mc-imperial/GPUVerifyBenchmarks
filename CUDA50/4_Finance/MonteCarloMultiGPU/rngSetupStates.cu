//pass
//--gridDim=256              --blockDim=256

__global__ void rngSetupStates(
    curandState *rngState,
    unsigned long long seed,
    unsigned long long offset)
{
    // determine global thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets the same seed, a different
    // sequence number. A different offset is used for
    // each device.
    curand_init(seed, tid, offset, &rngState[tid]);
}
