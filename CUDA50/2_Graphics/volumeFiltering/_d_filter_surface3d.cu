//pass
//--gridDim=[1,1,32]     --blockDim=[32,32]

//REQUIRES: SURFACE

#include "common.h"

texture<VolumeType, 3, VolumeTypeInfo<VolumeType>::readMode>  volumeTexIn;
surface<void,  3>                                             volumeTexOut;

__constant__ float4 c_filterData[VOLUMEFILTER_MAXWEIGHTS];
  
__global__ void
d_filter_surface3d(int filterSize, float filter_offset,
                   cudaExtent volumeSize)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
    {
        return;
    }

    float filtered = 0;
    float4 basecoord = make_float4(x,y,z,0);

    for (int i = 0; i < filterSize; i++)
    {
        float4 coord = basecoord + c_filterData[i];
        filtered  += tex3D(volumeTexIn,coord.x,coord.y,coord.z) * c_filterData[i].w;
    }

    filtered    += filter_offset;

    VolumeType output = VolumeTypeInfo<VolumeType>::convert(filtered);

    // surface writes need byte offsets for x!
    surf3Dwrite(output,volumeTexOut,x * sizeof(VolumeType),y,z);

}
