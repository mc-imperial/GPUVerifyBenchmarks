//pass
//--gridDim=40 --blockDim=256

typedef unsigned int uint;
__device__ float length(float2);

struct Ray
{
    float3 origin;
    float2 dir;
    int    length;
    float  oneOverLength;
};

__device__ static __attribute__((always_inline)) float2 getLocation(const Ray, int);
__device__ static __attribute__((always_inline)) float getAngle(const Ray, float2, float);

texture<float, 2, cudaReadModeElementType> g_HeightFieldTex;

__device__ static __attribute__((always_inline)) float2 getLocation(const Ray ray, int i)
{
    float step = i * ray.oneOverLength;
    return make_float2(ray.origin.x, ray.origin.y) + step * ray.dir;
}

__device__ static __attribute__((always_inline)) float getAngle(const Ray ray, float2 location, float height)
{
    float2 dir = location - make_float2(ray.origin.x, ray.origin.y);
    return atanf((height - ray.origin.z) / length(dir));
}

__global__ void computeAngles_kernel(const Ray ray, float *angles)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < ray.length)
    {
        float2 location = getLocation(ray, i + 1);
        float height = tex2D(g_HeightFieldTex, location.x, location.y);
        float angle = getAngle(ray, location, height);
        angles[i] = angle;
    }
}
