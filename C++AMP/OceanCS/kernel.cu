//pass
//--blockDim=[64,64] --gridDim=[8,8]

#include <cuda.h>

#define dimX 512
#define dimY 512

//--------------------------------------------------------------------------------------
// File: ocean_simulator.cpp
//
// Main class of ocean simulation
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#define dmap_dim 512

#define actual_dim dmap_dim
#define input_width (actual_dim + 4)
    // We use full sized data here. The value "output_width" should be actual_dim/2+1 though.
#define output_width actual_dim
#define output_height actual_dim
#define dtx_offset (actual_dim * actual_dim)
#define dty_offset (actual_dim * actual_dim * 2)


// Pre-FFT data preparation
__global__ void update_spectrum(
                     const float2* input_h0,
                     const float* input_omega,
                     float2* output_ht, 
                     unsigned int immutable_actualdim,
                     unsigned int immutable_inwidth,
                     unsigned int immutable_outwidth,
                     unsigned int immutable_outheight,
                     unsigned int immutable_dddressoffset,
                     unsigned int immutable_addressoffset,
                     float perframe_time)
{
    __requires(immutable_actualdim == 512 /*actual_dim*/);
    __requires(immutable_inwidth == 516 /*input_width*/);
    __requires(immutable_outwidth == 512 /*output_width*/);
    __requires(immutable_outheight == 512 /*output_height*/);
    __requires(immutable_dddressoffset == 512*512 /*dtx_offset*/);
    __requires(immutable_addressoffset == 512*512*2 /*dty_offset*/);
    {
        int in_index = (blockIdx.y * blockDim.y + threadIdx.y) * immutable_inwidth + (blockIdx.x * blockDim.x + threadIdx.x);
        int in_mindex = (immutable_actualdim - (blockIdx.y * blockDim.y + threadIdx.y)) * immutable_inwidth + (immutable_actualdim - (blockIdx.x * blockDim.x + threadIdx.x));
        int out_index = (blockIdx.y * blockDim.y + threadIdx.y) * immutable_outwidth + (blockIdx.x * blockDim.x + threadIdx.x);

        // H(0) -> H(t)
        float2 h0_k  = input_h0[in_index];
        float2 h0_mk = input_h0[in_mindex];
        float sin_v, cos_v;

      //sin_v = sincos(input_omega[in_index] * perframe_time, &cos_v);
        sin_v = sin(input_omega[in_index] * perframe_time);
        cos_v = cos(input_omega[in_index] * perframe_time);

        float2 ht;
        ht.x = (h0_k.x + h0_mk.x) * cos_v - (h0_k.y + h0_mk.y) * sin_v;
        ht.y = (h0_k.x - h0_mk.x) * sin_v + (h0_k.y - h0_mk.y) * cos_v;

        // H(t) -> Dx(t), Dy(t)
        float kx = (blockIdx.x * blockDim.x + threadIdx.x) - immutable_actualdim * 0.5f;
        float ky = (blockIdx.y * blockDim.y + threadIdx.y) - immutable_actualdim * 0.5f;
        float sqr_k = kx * kx + ky * ky;
        float rsqr_k = 0;
        if (sqr_k > 1e-12f) {
            rsqr_k = 1 / sqrt(sqr_k);
        }
        kx *= rsqr_k;
        ky *= rsqr_k;

        float2 dt_x;

        dt_x.x = ht.y * kx;
        dt_x.y = -ht.x * kx;

        float2 dt_y;

        dt_y.x = ht.y * ky;
        dt_y.y = -ht.x * ky;

        if (((blockIdx.x * blockDim.x + threadIdx.x) < immutable_outwidth) && 
            ((blockIdx.y * blockDim.y + threadIdx.y) < immutable_outwidth))
        {
            output_ht[out_index] = ht;
            output_ht[out_index + immutable_dddressoffset] = dt_x;
            output_ht[out_index + immutable_addressoffset] = dt_y;		
#ifdef MUTATION
            output_ht[out_index+1] = output_ht[out_index+1];
               /* BUGINJECT: ADD_ACCESS, UP */
#endif
        }    
    }
}

