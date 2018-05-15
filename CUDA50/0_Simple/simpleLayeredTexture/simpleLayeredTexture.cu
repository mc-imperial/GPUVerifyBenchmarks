//pass
//--gridDim=[64,64,1]      --blockDim=[8,8,1]

texture<float, cudaTextureType2DLayered> tex;

__global__ void
transformKernel(float *g_odata, int width, int height, int layer)
{
    __requires(width  == 64*8 /*gridDim.x * blockDim.x*/);
    __requires(height == 64*8 /*gridDim.y * blockDim.y*/);
    __requires(layer  == 1);

    // calculate this thread's data point
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // 0.5f offset and division are necessary to access the original data points
    // in the texture (such that bilinear interpolation will not be activated).
    // For details, see also CUDA Programming Guide, Appendix D
    float u = (x+0.5f) / (float) width;
    float v = (y+0.5f) / (float) height;

    // read from texture, do expected transformation and write to global memory
    g_odata[layer*width*height + y*width + x] = -tex2DLayered(tex, u, v, layer) + layer;
}
