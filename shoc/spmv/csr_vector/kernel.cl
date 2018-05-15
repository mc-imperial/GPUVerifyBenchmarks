//pass
//--global_size=32768 --local_size=128

#include "../common.h"

// ****************************************************************************
// Function: spmv_csr_vector_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a warp per row of the sparse
//   matrix; based on Bell (SC09) and Baskaran (IBM Tech Report)
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   vec: dense vector for multiplication
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation 
//   
// Returns:  nothing
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 28, 2010
//
// Modifications:
//
// ****************************************************************************
__kernel void 
spmv_csr_vector_kernel(__global const FPTYPE * restrict val, 
#ifdef USE_TEXTURE
                       image2d_t vec, 
#else
                       __global const FPTYPE * restrict vec, 
#endif
                       __global const int * restrict cols, 
                       __global const int * restrict rowDelimiters, 
                       const int dim, __global FPTYPE * restrict out) 
{
    __requires(dim == 1024);

    // Thread ID in block
    int t = get_local_id(0);
    // Thread ID within warp
    int id = t & (VECTOR_SIZE-1); 
    // One row per warp
    int vecsPerBlock = get_local_size(0) / VECTOR_SIZE;
    int myRow = (get_group_id(0) * vecsPerBlock) + (t / VECTOR_SIZE);

    __local volatile FPTYPE partialSums[128]; 
    partialSums[t] = 0; 

    if (myRow < dim) 
    {
        int vecStart = rowDelimiters[myRow];
        int vecEnd = rowDelimiters[myRow+1];
        FPTYPE mySum = 0;
        for (int j= vecStart + id; j < vecEnd; 
             j+=VECTOR_SIZE) 
        {
            int col = cols[j]; 
#ifdef USE_TEXTURE
            mySum += val[j] * texFetch(vec,col);
#else
            mySum += val[j] * vec[col];
#endif
        }

        partialSums[t] = mySum;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce partial sums
        // Needs to be modified if there is a change in vector
        // length
        if (id < 16) partialSums[t] += partialSums[t+16]; 
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id <  8) partialSums[t] += partialSums[t+ 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id <  4) partialSums[t] += partialSums[t+ 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id <  2) partialSums[t] += partialSums[t+ 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id <  1) partialSums[t] += partialSums[t+ 1];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Write result 
        if (id == 0) 
        {
            out[myRow] = partialSums[t]; 
        }
    }
}
