template<int i>  __device__ static __attribute__((always_inline)) float convolutionRow(float x, float y);
template<>       __device__ static __attribute__((always_inline)) float convolutionRow<-1>(float x, float y);
template<int i>  __device__ static __attribute__((always_inline)) float convolutionColumn(float x, float y);
template<>       __device__ static __attribute__((always_inline)) float convolutionColumn<-1>(float x, float y);

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
texture<float, 2, cudaReadModeElementType> texSrc;
__constant__ float c_Kernel[KERNEL_LENGTH];

template<int i> __device__ static __attribute__((always_inline)) float convolutionRow(float x, float y)
{
    return
        tex2D(texSrc, x + (float)(KERNEL_RADIUS - i), y) * c_Kernel[i]
        + convolutionRow<i - 1>(x, y);
}

template<> __device__ static __attribute__((always_inline)) float convolutionRow<-1>(float x, float y)
{
    return 0;
}

template<int i> __device__ static __attribute__((always_inline)) float convolutionColumn(float x, float y)
{
    return
        tex2D(texSrc, x, y + (float)(KERNEL_RADIUS - i)) * c_Kernel[i]
        + convolutionColumn<i - 1>(x, y);
}

template<> __device__ static __attribute__((always_inline)) float convolutionColumn<-1>(float x, float y)
{
    return 0;
}
