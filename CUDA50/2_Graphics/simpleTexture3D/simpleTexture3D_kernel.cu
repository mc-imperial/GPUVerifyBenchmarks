//pass
//--gridDim=[32,32,1]      --blockDim=[16,16,1]

typedef unsigned char uchar;
texture<uchar, 3, cudaReadModeNormalizedFloat> tex;  // 3D texture

typedef unsigned int uint;
#define __umul24(x,y) (x*y)

__global__ void
d_render(uint *d_output, uint imageW, uint imageH, float w)
{
    __requires(imageW == 32*16 /*gridDim.x*blockDim.x*/);

    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    float u = x / (float) imageW;
    float v = y / (float) imageH;
    // read from 3D texture
    float voxel = tex3D(tex, u, v, w);

    if ((x < imageW) && (y < imageH))
    {
        // write output color
        uint i = __umul24(y, imageW) + x;
        d_output[i] = voxel*255;
    }
}
