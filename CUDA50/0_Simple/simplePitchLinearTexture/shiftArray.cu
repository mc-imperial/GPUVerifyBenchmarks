//pass
//--gridDim=[128,128,1]    --blockDim=[16,16,1]

texture<float, 2, cudaReadModeElementType> texRefArray;

__global__ void shiftArray(float *odata,
                           int pitch,
                           int width,
                           int height,
                           int shiftX,
                           int shiftY)
{
    __requires(pitch == 2048);
    __requires(width == 2048);
    __requires(height == 2048);

    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    odata[yid * pitch + xid] = tex2D(texRefArray,
                                     (xid + shiftX) / (float) width,
                                     (yid + shiftY) / (float) height);
}
