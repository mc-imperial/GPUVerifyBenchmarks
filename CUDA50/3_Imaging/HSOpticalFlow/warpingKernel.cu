//pass
//--gridDim=[10,40]      --blockDim=[32,6]

texture<float, 2, cudaReadModeElementType> texToWarp;

__global__ void WarpingKernel(int width, int height, int stride,
                              const float *u, const float *v, float *out)
{
    __requires(width == 320);
    __requires(height == 240);
    __requires(stride == 320);
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height) return;

    float x = ((float)ix + u[pos] + 0.5f) / (float)width;
    float y = ((float)iy + v[pos] + 0.5f) / (float)height;

    out[pos] = tex2D(texToWarp, x, y);
}
