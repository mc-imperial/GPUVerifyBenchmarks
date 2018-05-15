//pass
//--gridDim=[32,32,1] --blockDim=[8,8,1]

__device__ static __attribute__((always_inline)) float2 conjugate(float2 arg);
__device__ static __attribute__((always_inline)) float2 complex_exp(float arg);
__device__ static __attribute__((always_inline)) float2 complex_add(float2 a, float2 b);
__device__ static __attribute__((always_inline)) float2 complex_mult(float2 ab, float2 cd);

__device__ static __attribute__((always_inline))
float2 conjugate(float2 arg)
{
    return make_float2(arg.x, -arg.y);
}

__device__ static __attribute__((always_inline))
float2 complex_exp(float arg)
{
    return make_float2(cosf(arg), sinf(arg));
}

__device__ static __attribute__((always_inline))
float2 complex_add(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ static __attribute__((always_inline))
float2 complex_mult(float2 ab, float2 cd)
{
    return make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
}

__global__ void generateSpectrumKernel(float2 *h0,
                                       float2 *ht,
                                       unsigned int in_width,
                                       unsigned int out_width,
                                       unsigned int out_height,
                                       float t,
                                       float patchSize)
{
    __requires(out_width == 256);
    __requires(out_height == 256);

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int in_index = y*in_width+x;
    unsigned int in_mindex = (out_height - y)*in_width + (out_width - x); // mirrored
    unsigned int out_index = y*out_width+x;

    // calculate wave vector
    float2 k;
    k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
    k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

    // calculate dispersion w(k)
    float k_len = sqrtf(k.x*k.x + k.y*k.y);
    float w = sqrtf(9.81f * k_len);

    if ((x < out_width) && (y < out_height))
    {
        float2 h0_k = h0[in_index];
        float2 h0_mk = h0[in_mindex];

        // output frequency-space complex values
        ht[out_index] = complex_add(complex_mult(h0_k, complex_exp(w * t)), complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
        //ht[out_index] = h0_k;
    }
}
