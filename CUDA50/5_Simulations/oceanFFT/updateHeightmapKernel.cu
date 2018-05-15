//pass
//--gridDim=[32,32,1] --blockDim=[8,8,1]

__global__ void updateHeightmapKernel(float  *heightMap,
                                      float2 *ht,
                                      unsigned int width)
{
    __requires(width == 256);

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int i = y*width+x;

    // cos(pi * (m1 + m2))
    float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

    heightMap[i] = ht[i].x * sign_correction;
}
