#define T float
#define T2 float2

#ifndef M_PI
# define M_PI 3.14159265358979323846f
#endif

#ifndef M_SQRT1_2
# define M_SQRT1_2      0.70710678118654752440f
#endif

#define exp_1_8   (T2)(  1, -1 )//requires post-multiply by 1/sqrt(2)
#define exp_1_4   (T2)(  0, -1 )
#define exp_3_8   (T2)( -1, -1 )//requires post-multiply by 1/sqrt(2)

#define iexp_1_8   (T2)(  1, 1 )//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   (T2)(  0, 1 )
#define iexp_3_8   (T2)( -1, 1 )//requires post-multiply by 1/sqrt(2)
 
static __attribute__((always_inline)) void globalLoads8(T2 *data, __global T2 *in, int stride){
    for( int i = 0; i < 8; i++ )
        data[i] = in[i*stride];
}


static __attribute__((always_inline)) void globalStores8(T2 *data, __global T2 *out, int stride){
    int reversed[] = {0,4,2,6,1,5,3,7};

//#pragma unroll
    for( int i = 0; i < 8; i++ )
        out[i*stride] = data[reversed[i]];
}


static __attribute__((always_inline)) void storex8( T2 *a, __local T *x, int sx ) {
    int reversed[] = {0,4,2,6,1,5,3,7};

//#pragma unroll
    for( int i = 0;
         __global_invariant(__write_implies(x, (__write_offset_bytes(x)/sizeof(T) - get_local_id(0))/sx < 8)),
         __global_invariant(__write_implies(x, (__write_offset_bytes(x)/sizeof(T) - get_local_id(0))%sx == 0)),
         i < 8; i++ )
        x[i*sx] = a[reversed[i]].x;
}

static __attribute__((always_inline)) void storey8( T2 *a, __local T *x, int sx ) {
    int reversed[] = {0,4,2,6,1,5,3,7};

//#pragma unroll
    for( int i = 0;
         __global_invariant(__write_implies(x, (__write_offset_bytes(x)/sizeof(T) - get_local_id(0))/sx < 8)),
         __global_invariant(__write_implies(x, (__write_offset_bytes(x)/sizeof(T) - get_local_id(0))%sx == 0)),
        i < 8; i++ )
        x[i*sx] = a[reversed[i]].y;
}


static __attribute__((always_inline)) void loadx8( T2 *a, __local T *x, int sx ) {
    for( int i = 0; i < 8; i++ )
        a[i].x = x[i*sx];
}

static __attribute__((always_inline)) void loady8( T2 *a, __local T *x, int sx ) {
    for( int i = 0; i < 8; i++ )
        a[i].y = x[i*sx];
}


#define transpose( a, s, ds, l, dl, sync )                              \
{                                                                       \
    storex8( a, s, ds );  if( (sync)&8 ) barrier(CLK_LOCAL_MEM_FENCE);  \
    loadx8 ( a, l, dl );  if( (sync)&4 ) barrier(CLK_LOCAL_MEM_FENCE);  \
    storey8( a, s, ds );  if( (sync)&2 ) barrier(CLK_LOCAL_MEM_FENCE);  \
    loady8 ( a, l, dl );  if( (sync)&1 ) barrier(CLK_LOCAL_MEM_FENCE);  \
}

static __attribute__((always_inline)) T2 exp_i( T phi ) {
//#ifdef USE_NATIVE
//    return (T2)( native_cos(phi), native_sin(phi) );
//#else
    return (T2)( cos(phi), sin(phi) );
//#endif    
}

static __attribute__((always_inline)) T2 cmplx_mul( T2 a, T2 b ) { return (T2)( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
static __attribute__((always_inline)) T2 cm_fl_mul( T2 a, T  b ) { return (T2)( b*a.x, b*a.y ); }
static __attribute__((always_inline)) T2 cmplx_add( T2 a, T2 b ) { return (T2)( a.x + b.x, a.y + b.y ); }
static __attribute__((always_inline)) T2 cmplx_sub( T2 a, T2 b ) { return (T2)( a.x - b.x, a.y - b.y ); }


#define twiddle8(a, i, n )                                              \
{                                                                       \
    int reversed8[] = {0,4,2,6,1,5,3,7};                                \
    for( int j = 1; j < 8; j++ ){                                       \
        a[j] = cmplx_mul( a[j],exp_i((-2*M_PI*reversed8[j]/(n))*(i)) ); \
    }                                                                   \
}

#define FFT2(a0, a1)                            \
{                                               \
    T2 c0 = *a0;                           \
    *a0 = cmplx_add(c0,*a1);                    \
    *a1 = cmplx_sub(c0,*a1);                    \
}

#define FFT4(a0, a1, a2, a3)                    \
{                                               \
    FFT2( a0, a2 );                             \
    FFT2( a1, a3 );                             \
    *a3 = cmplx_mul(*a3,exp_1_4);               \
    FFT2( a0, a1 );                             \
    FFT2( a2, a3 );                             \
}

#define FFT8(a)                                                 \
{                                                               \
    FFT2( &a[0], &a[4] );                                       \
    FFT2( &a[1], &a[5] );                                       \
    FFT2( &a[2], &a[6] );                                       \
    FFT2( &a[3], &a[7] );                                       \
                                                                \
    a[5] = cm_fl_mul( cmplx_mul(a[5],exp_1_8) , M_SQRT1_2 );    \
    a[6] =  cmplx_mul( a[6] , exp_1_4);                         \
    a[7] = cm_fl_mul( cmplx_mul(a[7],exp_3_8) , M_SQRT1_2 );    \
                                                                \
    FFT4( &a[0], &a[1], &a[2], &a[3] );                         \
    FFT4( &a[4], &a[5], &a[6], &a[7] );                         \
}

#define itwiddle8( a, i, n )                                            \
{                                                                       \
    int reversed8[] = {0,4,2,6,1,5,3,7};                                \
    for( int j = 1; j < 8; j++ )                                        \
        a[j] = cmplx_mul(a[j] , exp_i((2*M_PI*reversed8[j]/(n))*(i)) ); \
}

#define IFFT2 FFT2

#define IFFT4( a0, a1, a2, a3 )                 \
{                                               \
    IFFT2( a0, a2 );                            \
    IFFT2( a1, a3 );                            \
    *a3 = cmplx_mul(*a3 , iexp_1_4);            \
    IFFT2( a0, a1 );                            \
    IFFT2( a2, a3);                             \
}

#define IFFT8( a )                                              \
{                                                               \
    IFFT2( &a[0], &a[4] );                                      \
    IFFT2( &a[1], &a[5] );                                      \
    IFFT2( &a[2], &a[6] );                                      \
    IFFT2( &a[3], &a[7] );                                      \
                                                                \
    a[5] = cm_fl_mul( cmplx_mul(a[5],iexp_1_8) , M_SQRT1_2 );   \
    a[6] = cmplx_mul( a[6] , iexp_1_4);                         \
    a[7] = cm_fl_mul( cmplx_mul(a[7],iexp_3_8) , M_SQRT1_2 );   \
                                                                \
    IFFT4( &a[0], &a[1], &a[2], &a[3] );                        \
    IFFT4( &a[4], &a[5], &a[6], &a[7] );                        \
}
