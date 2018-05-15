// --------------------------------------------------------------------------
// Types
// --------------------------------------------------------------------------

// From volume.h
typedef unsigned char VolumeType;

template< typename T >
struct VolumeTypeInfo
{};

template< >
struct VolumeTypeInfo<unsigned char>
{
    static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;
    static __device__ __attribute__((always_inline)) unsigned char convert(float sampled)
    {
        return (unsigned char)(saturate(sampled) * 255.0);
    }
};

template< >
struct VolumeTypeInfo<unsigned short>
{
    static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;
    static __device__ __attribute((always_inline)) unsigned short convert(float sampled)
    {
        return (unsigned short)(saturate(sampled) * 65535.0);
    }
};

template< >
struct VolumeTypeInfo<float>
{
    static const cudaTextureReadMode readMode = cudaReadModeElementType;
    static __device__ __attribute__((always_inline)) float convert(float sampled)
    {
        return sampled;
    }
};

// From volumeRender_kernel.cu
struct Ray
{
    float3 o;    // origin
    float3 d;    // direction
};

typedef struct
{
    float4 m[3];
} float3x4;

#define VOLUMERENDER_TF_PREINTRAY     4

enum TFMode
{
    TF_SINGLE_1D = 0,         // single 1D TF for everything
    TF_LAYERED_2D_PREINT = 1, // layered 2D TF uses pre-integration
    TF_LAYERED_2D = 2,        // layered 2D TF without pre-integration behavior
};

// --------------------------------------------------------------------------
// Functions
// --------------------------------------------------------------------------

__device__ static __attribute__((always_inline))
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

__device__ static __attribute__((always_inline))
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ static __attribute__((always_inline))
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin  = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

typedef unsigned int uint;
__device__ static __attribute__((always_inline))
uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

// 3D texture
texture<VolumeType, 3, VolumeTypeInfo<VolumeType>::readMode>  volumeTex;
// 1D transfer function texture
texture<float4, 1, cudaReadModeElementType>           transferTex;
// 1D transfer integration texture
texture<float4, 1, cudaReadModeElementType>           transferIntegrateTex;
#ifdef IMPLEMENT_SURFACE
surface<void, 1>                                      transferIntegrateSurf;
#endif
// 2D layered preintegrated transfer function texture
texture<float4, cudaTextureType2DLayered,cudaReadModeElementType>   transferLayerPreintTex;
#ifdef IMPLEMENT_SURFACE
surface<void, cudaSurfaceType2DLayered>                             transferLayerPreintSurf;
#endif

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

template <int TFMODE >
__device__ static __attribute__((always_inline))
void d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, float transferWeight = 0.0f)
{
    const float rayscale =  float(TFMODE != TF_SINGLE_1D ? VOLUMERENDER_TF_PREINTRAY : 1);
    const int maxSteps = 512;
    const float tstep = 0.01f * rayscale;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    density *= rayscale;

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;
#ifdef VOLUMERENDER_RANDSIZE
    float  offset = (tex2D(rayTex,u,v));
    pos += step * offset;
#endif
    float lastsample = 0;

    //lastsample = (lastsample-transferOffset)*transferScale;
    for (int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float3 coord = make_float3(pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
        float sample = tex3D(volumeTex, coord.x, coord.y, coord.z);
        //sample = (sample-transferOffset)*transferScale;
        //sample *= 64.0f;    // scale for 10-bit data

        // lookup in transfer function texture
        float4 col;
        int tfid = (pos.x < 0);

        if (TFMODE != TF_SINGLE_1D)
        {
            col = tex2DLayered(transferLayerPreintTex, sample, TFMODE==TF_LAYERED_2D ? sample : lastsample, tfid);
            col.w *= density;
            lastsample = sample;
        }
        else
        {
            col = tex1D(transferTex, sample);
            col.w *= 0;
        }

        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);


        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}
