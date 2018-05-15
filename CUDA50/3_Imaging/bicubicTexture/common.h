                           __device__ static __attribute__((always_inline)) float w0(float a);
                           __device__ static __attribute__((always_inline)) float w1(float a);
                           __device__ static __attribute__((always_inline)) float w2(float a);
                           __device__ static __attribute__((always_inline)) float w3(float a);
                           __device__ static __attribute__((always_inline)) float g0(float a);
                           __device__ static __attribute__((always_inline)) float g1(float a);
                           __device__ static __attribute__((always_inline)) float h0(float a);
                           __device__ static __attribute__((always_inline)) float h1(float a);
template<class T>          __device__ static __attribute__((always_inline)) T cubicFilter(float x, T c0, T c1, T c2, T c3);
template<class T, class R> __device__ static __attribute__((always_inline)) R tex2DBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y);
template<class T, class R> __device__ static __attribute__((always_inline)) R tex2DFastBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y);
template<class T, class R> __device__ static __attribute__((always_inline)) R tex2DBilinear(const texture<T, 2, cudaReadModeNormalizedFloat> tex, float x, float y);
                           __device__ static __attribute__((always_inline)) float catrom_w0(float a);
                           __device__ static __attribute__((always_inline)) float catrom_w1(float a);
                           __device__ static __attribute__((always_inline)) float catrom_w2(float a);
                           __device__ static __attribute__((always_inline)) float catrom_w3(float a);
template<class T>          __device__ static __attribute__((always_inline)) T catRomFilter(float x, T c0, T c1, T c2, T c3);
template<class T, class R> __device__ static __attribute__((always_inline)) R tex2DCatRom(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y);

typedef unsigned char uchar;
typedef unsigned int  uint;
texture<uchar, 2, cudaReadModeNormalizedFloat> tex;
texture<uchar, 2, cudaReadModeElementType> tex2;    // need to use cudaReadModeElementType for tex2Dgather

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__device__ static inline __attribute__((always_inline))
float w0(float a)
{
    //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__device__ static inline __attribute__((always_inline))
float w1(float a)
{
    //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__device__ static inline __attribute__((always_inline))
float w2(float a)
{
    //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__device__ static inline __attribute__((always_inline))
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ static inline __attribute__((always_inline)) float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ static inline __attribute__((always_inline)) float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ static inline __attribute__((always_inline)) float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ static inline __attribute__((always_inline)) float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

// filter 4 values using cubic splines
template<class T>
__device__ static inline __attribute__((always_inline))
T cubicFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

// slow but precise bicubic lookup using 16 texture lookups
template<class T, class R>  // texture data type, return type
__device__ static inline __attribute__((always_inline))
R tex2DBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter<R>(fy,
                          cubicFilter<R>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
                         );
}

// fast bicubic texture lookup using 4 bilinear lookups
// assumes texture is set to non-normalized coordinates, point sampling
template<class T, class R>  // texture data type, return type
__device__ static inline __attribute__((always_inline))
R tex2DFastBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    R r = g0(fy) * (g0x * tex2D(texref, px + h0x, py + h0y)   +
                    g1x * tex2D(texref, px + h1x, py + h0y)) +
          g1(fy) * (g0x * tex2D(texref, px + h0x, py + h1y)   +
                    g1x * tex2D(texref, px + h1x, py + h1y));
    return r;
}

// higher-precision 2D bilinear lookup
template<class T, class R>  // texture data type, return type
__device__ static inline __attribute__((always_inline))
R tex2DBilinear(const texture<T, 2, cudaReadModeNormalizedFloat> tex, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floorf(x);   // integer position
    float py = floorf(y);
    float fx = x - px;      // fractional position
    float fy = y - py;
    px += 0.5f;
    py += 0.5f;

    return lerp(lerp(tex2D(tex, px, py),        tex2D(tex, px + 1.0f, py), fx),
                lerp(tex2D(tex, px, py + 1.0f), tex2D(tex, px + 1.0f, py + 1.0f), fx), fy);
}

// Catmull-Rom interpolation

__device__ static inline __attribute__((always_inline))
float catrom_w0(float a)
{
    //return -0.5f*a + a*a - 0.5f*a*a*a;
    return a*(-0.5f + a*(1.0f - 0.5f*a));
}

__device__ static inline __attribute__((always_inline))
float catrom_w1(float a)
{
    //return 1.0f - 2.5f*a*a + 1.5f*a*a*a;
    return 1.0f + a*a*(-2.5f + 1.5f*a);
}

__device__ static inline __attribute__((always_inline))
float catrom_w2(float a)
{
    //return 0.5f*a + 2.0f*a*a - 1.5f*a*a*a;
    return a*(0.5f + a*(2.0f - 1.5f*a));
}

__device__ static inline __attribute__((always_inline))
float catrom_w3(float a)
{
    //return -0.5f*a*a + 0.5f*a*a*a;
    return a*a*(-0.5f + 0.5f*a);
}

template<class T>
__device__ static inline __attribute__((always_inline))
T catRomFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * catrom_w0(x);
    r += c1 * catrom_w1(x);
    r += c2 * catrom_w2(x);
    r += c3 * catrom_w3(x);
    return r;
}

// Note - can't use bilinear trick here because of negative lobes
template<class T, class R>  // texture data type, return type
__device__ static inline __attribute__((always_inline))
R tex2DCatRom(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return catRomFilter<R>(fy,
                           catRomFilter<R>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
                           catRomFilter<R>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
                           catRomFilter<R>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
                           catRomFilter<R>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
                          );
}
