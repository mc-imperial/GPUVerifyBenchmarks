//pass
//--num_groups=8 --local_size=128

#include "../common.h"

// ****************************************************************************
// Function: spmv_ellpackr_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the ELLPACK-R data storage format; based on Vazquez et al (Univ. of
//   Almeria Tech Report 2009)
//
// Arguments:
//   val: array holding the non-zero values for the matrix in column
//   vec: dense vector for multiplication
//   major format and padded with zeros up to the length of longest row
//   cols: array of column indices for each element of the sparse matrix
//   rowLengths: array storing the length of each row of the sparse matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation 
//   
// Returns:  nothing directly
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 29, 2010
//
// Modifications:
//
// ****************************************************************************
__kernel void
spmv_ellpackr_kernel(__global const FPTYPE * restrict val, 
#ifdef USE_TEXTURE
                     image2d_t vec, 
#else
                     __global const  FPTYPE * restrict vec,                      
#endif
                     __global const int * restrict cols, 
                     __global const int * restrict rowLengths, 
                     const int dim, __global FPTYPE * restrict out) 
{
    int t = get_global_id(0); 

    if (t < dim) 
    {
        FPTYPE result = 0.0;
        int max = rowLengths[t]; 
        for (int i = 0; i < max; i++) 
        {
            int ind = i * dim + t; 
#ifdef USE_TEXTURE
	          result += val[ind] * texFetch(vec,cols[ind]);
#else
	          result += val[ind] * vec[cols[ind]];
#endif
        }
        out[t] = result;
    }
}
