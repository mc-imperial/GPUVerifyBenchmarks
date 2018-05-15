//pass
//--gridDim=[8,8,1]        --blockDim=[8,8,1]

texture<float, cudaTextureTypeCubemap> tex;

__global__ void
transformKernel(float *g_odata, int width)
{
    __requires(width == 8*8 /*gridDim.x * blockDim.x*/);

    // calculate this thread's data point
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // 0.5f offset and division are necessary to access the original data points
    // in the texture (such that bilinear interpolation will not be activated).
    // For details, see also CUDA Programming Guide, Appendix D

    float u = ((x+0.5f) / (float) width) * 2.f - 1.f;
    float v = ((y+0.5f) / (float) width) * 2.f - 1.f;

    float cx, cy, cz;

    for (unsigned int face = 0; face < 6; face ++)
    {
        //Layer 0 is positive X face
        if (face == 0)
        {
            cx = 1;
            cy = -v;
            cz = -u;
        }
        //Layer 1 is negative X face
        else if (face == 1)
        {
            cx = -1;
            cy = -v;
            cz = u;
        }
        //Layer 2 is positive Y face
        else if (face == 2)
        {
            cx = u;
            cy = 1;
            cz = v;
        }
        //Layer 3 is negative Y face
        else if (face == 3)
        {
            cx = u;
            cy = -1;
            cz = -v;
        }
        //Layer 4 is positive Z face
        else if (face == 4)
        {
            cx = u;
            cy = -v;
            cz = 1;
        }
        //Layer 4 is negative Z face
        else if (face == 5)
        {
            cx = -u;
            cy = -v;
            cz = -1;
        }

        // read from texture, do expected transformation and write to global memory
        g_odata[face*width*width + y*width + x] = -texCubemap(tex, cx, cy, cz);
    }
}
