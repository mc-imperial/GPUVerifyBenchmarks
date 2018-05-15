//pass
//--gridDim=[32,32] --blockDim=[16,16]

typedef unsigned int  uint;
typedef unsigned char uchar;
texture<uint2, 2, cudaReadModeElementType> atlasTexture;

__device__ static __attribute__((always_inline)) cudaTextureObject_t decodeTextureObject(uint2 obj)
{
    return (((cudaTextureObject_t)obj.x) | ((cudaTextureObject_t)obj.y) << 32);
}

__device__ static __attribute__((always_inline)) uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

__global__ void
d_render(uchar4 *d_output, uint imageW, uint imageH, float lod)
{
    __requires(imageW == 16*32 /*blockDim.x * gridDim.x*/);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float) imageW;
    float v = y / (float) imageH;

    if ((x < imageW) && (y < imageH))
    {
        // read from 2D atlas texture and decode texture object
        uint2 texCoded = tex2D(atlasTexture, u, v);
        cudaTextureObject_t tex = decodeTextureObject(texCoded);

        // read from cuda texture object, use template to specify what data will be
        // returned. tex2DLod allows us to pass the lod (mip map level) directly.
        // There is other functions with CUDA 5, e.g. tex2DGrad,    that allow you
        // to pass derivatives to perform automatic mipmap/anisotropic filtering.
        float4 color = tex2DLod<float4>(tex, u, 1-v, lod);
        // In our sample tex is always valid, but for something like your own
        // sparse texturing you would need to make sure to handle the zero case.

        // write output color
        uint i = y * imageW + x;
        d_output[i] = to_uchar4(color * 255.0);
    }
}
