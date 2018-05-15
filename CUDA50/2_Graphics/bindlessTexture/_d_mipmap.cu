//pass
//--gridDim=[19,15] --blockDim=[16,16]

//REQUIRES:SURFACE

typedef unsigned int uchar;
typedef unsigned int uint;

__device__ float4 fminf(float4, float4);

__device__ static __attribute__((always_inline)) uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

__global__ void
d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH)
{
    __requires(imageW == 16*32 /*blockDim.x * gridDim.x*/);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0/float(imageW);
    float py = 1.0/float(imageH);

    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        float4 color = 
            (tex2D<float4>(mipInput,(x + 0) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput,(x + 1) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput,(x + 1) * px, (y + 1) * py)) +
            (tex2D<float4>(mipInput,(x + 0) * px, (y + 1) * py));

        color /= 4.0;
        color *= 255.0;
        color = fminf(color,make_float4(255.0));

        surf2Dwrite(to_uchar4(color),mipOutput,x * sizeof(uchar4),y);
    }
}
