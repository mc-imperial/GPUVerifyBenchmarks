//pass
//--gridDim=40 --blockDim=256

typedef unsigned char Bool;
typedef unsigned int uint;

__global__ void computeVisibilities_kernel(const float *angles,
                                           const float *scannedAngles,
                                           int numAngles,
                                           Bool *visibilities)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numAngles)
    {
        visibilities[i] = scannedAngles[i] <= angles[i];
    }
}
