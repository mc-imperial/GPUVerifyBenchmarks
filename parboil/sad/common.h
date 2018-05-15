/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#define MAX_POS_PADDED 1096
#define MAX_POS 1089
#define POS_PER_THREAD 18
#define THREADS_W 1
#define THREADS_H 1
#define SEARCH_RANGE 16
#define SEARCH_DIMENSION (2*SEARCH_RANGE+1)
#define CEIL(x,y) (((x) + ((y) - 1)) / (y))
#define CEIL_POS CEIL(MAX_POS, POS_PER_THREAD)


/* The compute kernel. */
/* The macros THREADS_W and THREADS_H specify the width and height of the
 * area to be processed by one thread, measured in 4-by-4 pixel blocks.
 * Larger numbers mean more computation per thread block.
 *
 * The macro POS_PER_THREAD specifies the number of search positions for which
 * an SAD is computed.  A larger value indicates more computation per thread,
 * and fewer threads per thread block.  It must be a multiple of 3 and also
 * must be at most 33 because the loop to copy from shared memory uses
 * 32 threads per 4-by-4 pixel block.
 *
 */
 
// AMD OpenCL fails UINT_CUDA_V
#if 1
  #define SHORT2_V 1
  #define UINT_CUDA_V 0
#else
  #define SHORT2_V 0
  #define UINT_CUDA_V 1
#endif

// Either works
#if 0
  #define VEC_LOAD 1
  #define CONSTR_LOAD 0
#else
  #define VEC_LOAD 0
  #define CONSTR_LOAD 1
#endif

// CAST_STORE is only method that works for all implementations of OpenCL tested
#if 0
  #define VEC_STORE 1
  #define CAST_STORE 0
  #define SCALAR_STORE 0
#elif 1
  #define VEC_STORE 0
  #define CAST_STORE 1
  #define SCALAR_STORE 0
#else
  #define VEC_STORE 0
  #define CAST_STORE 0
  #define SCALAR_STORE 1
#endif

