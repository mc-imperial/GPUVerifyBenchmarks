//pass
//--gridDim=[64,64,1]      --blockDim=[8,8,1]

texture<float, 2, cudaReadModeElementType> tex;

__global__ void transformKernel(float *outputData,
                                int width,
                                int height,
                                float theta)
{
    __requires(width == 64*8 /*gridDim.x * blockDim.x*/);

    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = x / (float) width;
    float v = y / (float) height;

    // transform coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u*cosf(theta) - v*sinf(theta) + 0.5f;
    float tv = v*cosf(theta) + u*sinf(theta) + 0.5f;

    // read from texture and write to global memory
    outputData[y*width + x] = tex2D(tex, tu, tv);
}
