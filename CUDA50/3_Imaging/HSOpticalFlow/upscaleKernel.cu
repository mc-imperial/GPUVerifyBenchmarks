//pass
//--gridDim=[10,30]      --blockDim=[32,8]

texture<float, 2, cudaReadModeElementType> texCoarse;

__global__ void UpscaleKernel(int width, int height, int stride, float scale, float *out)
{
    __requires(width == 320);
    __requires(height == 240);
    __requires(stride == 320);
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) return;

    float x = ((float)ix + 0.5f) / (float)width;
    float y = ((float)iy + 0.5f) / (float)height;

    // exploit hardware interpolation
    // and scale interpolated vector to match next pyramid level resolution
    out[ix + iy * stride] = tex2D(texCoarse, x, y) * scale;
}
