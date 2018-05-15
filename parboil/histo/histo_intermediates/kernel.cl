//pass
//--local_size=[498] --num_groups=[65]

#include "../common.h"

static __attribute__((always_inline))
void calculateBin (
        __const unsigned int bin,
        __global uchar4 *sm_mapping)
{
        unsigned char offset  =  bin        %   4;
        unsigned char indexlo = (bin >>  2) % 256;
        unsigned char indexhi = (bin >> 10) %  KB;
        unsigned char block   =  bin / BINS_PER_BLOCK;

        offset *= 8;

        uchar4 sm;
        sm.x = block;
        sm.y = indexhi;
        sm.z = indexlo;
        sm.w = offset;

        *sm_mapping = sm;
}

__kernel void histo_intermediates_kernel (
        __global uint2 *input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch,
        __global uchar4 *sm_mappings)
{
        __requires(width == 996);
        int threadIdxx = get_local_id(0);
        int blockDimx = get_local_size(0);
        unsigned int line = UNROLL * (get_group_id(0));// 16 is the unroll factor;

        __global uint2 *load_bin = input + line * input_pitch + threadIdxx;

        unsigned int store = line * width + threadIdxx;
        bool skip = (width % 2) && (threadIdxx == (blockDimx - 1));

        #pragma unroll
        for (int i = 0;
             __global_invariant(__write_implies(sm_mappings,
                             (       ((__write_offset_bytes(sm_mappings)/sizeof(uchar4) - (line * width + threadIdxx))%width == 0) &
                                     ((__write_offset_bytes(sm_mappings)/sizeof(uchar4) - (line * width + threadIdxx))/width < UNROLL))
                    |
                    __implies(!skip, ((__write_offset_bytes(sm_mappings)/sizeof(uchar4) - (line * width + threadIdxx + blockDimx))%width == 0) &
                                     ((__write_offset_bytes(sm_mappings)/sizeof(uchar4) - (line * width + threadIdxx + blockDimx))/width < UNROLL)))),
             __invariant(store == line * width + threadIdxx + i * width),
             i < UNROLL; i++)
        {
                uint2 bin_value = *load_bin;

                calculateBin (
                        bin_value.x,
                        &sm_mappings[store]
                );

                if (!skip) calculateBin (
                        bin_value.y,
                        &sm_mappings[store + blockDimx]
                );

                load_bin += input_pitch;
                store += width;
        }
}
