//pass
//--gridDim=[10,30]      --blockDim=[32,8]

texture<float, 2, cudaReadModeElementType> texFine;

__global__ void DownscaleKernel(int width, int height, int stride, float *out)
{
    __requires(width == 320);
    __requires(height == 240);
    __requires(stride == 320);
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height)
    {
        return;
    }

    float dx = 1.0f/(float)width;
    float dy = 1.0f/(float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

    out[ix + iy * stride] = 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
                                     tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));
}
