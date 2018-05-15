//pass
//--global_size=[64,64] --local_size=[16,4]

#include "../common.h"

__kernel void sgemmNN( __global const FPTYPE *A, int lda,
                       __global const FPTYPE *B, int ldb,
                       __global FPTYPE *C, int ldc, int k,
                       FPTYPE alpha, FPTYPE beta )
{
  __requires(ldc == 256);

	const int inx = get_local_id(0);
	const int iny = get_local_id(1);
	const int ibx = get_group_id(0) * 64;
	const int iby = get_group_id(1) * 16;
	const int id = inx + iny*16;
	
        int i, j, ii, counter=0;

	A += ibx + id;

	B += inx + (iby+iny) * ldb;

	C += ibx + id  + (iby*ldc);
	
	FPTYPE c[16];
        for(i=0; i<16; ++i){
            c[i] = 0.0;
	}

       	__local FPTYPE bs[16][17];
	do
	{
		__private FPTYPE a[4];
		for(ii=0; ii<4; ++ii) { a[ii] = A[ii*lda]; }

		bs[inx][iny]    = B[0*ldb];
		bs[inx][iny+4]  = B[4*ldb];
		bs[inx][iny+8]  = B[8*ldb];
		bs[inx][iny+12] = B[12*ldb];
		barrier(CLK_LOCAL_MEM_FENCE);

		A += 4*lda;

		SAXPY( a[0], bs[0], c );	a[0] = A[0*lda];
		SAXPY( a[1], bs[1], c );	a[1] = A[1*lda];
		SAXPY( a[2], bs[2], c );	a[2] = A[2*lda];
		SAXPY( a[3], bs[3], c );	a[3] = A[3*lda];	
 
		A += 4*lda;
		SAXPY( a[0], bs[4], c );	a[0] = A[0*lda];
		SAXPY( a[1], bs[5], c );	a[1] = A[1*lda];
		SAXPY( a[2], bs[6], c );	a[2] = A[2*lda];
		SAXPY( a[3], bs[7], c );	a[3] = A[3*lda];
		
		A += 4*lda;
		SAXPY( a[0], bs[8], c );	a[0] = A[0*lda];
		SAXPY( a[1], bs[9], c );	a[1] = A[1*lda];
		SAXPY( a[2], bs[10], c );	a[2] = A[2*lda];
		SAXPY( a[3], bs[11], c );	a[3] = A[3*lda];
		
		A += 4*lda;
		SAXPY( a[0], bs[12], c );
		SAXPY( a[1], bs[13], c );
		SAXPY( a[2], bs[14], c );
		SAXPY( a[3], bs[15], c );

		B += 16;
	        counter += 16;
		barrier(CLK_LOCAL_MEM_FENCE);
	} while( counter < k );
	
	for( int i = 0;
         __global_invariant((__ptr_offset_bytes(C)/sizeof(FPTYPE) - (ibx + id + iby * ldc)) == ldc * i),
         __global_invariant(__write_implies(C, (__write_offset_bytes(C)/sizeof(FPTYPE) - (ibx + id + iby * ldc))/ldc < 16)),
         __global_invariant(__read_implies(C, (__read_offset_bytes(C)/sizeof(FPTYPE) - (ibx + id + iby * ldc))/ldc < 16)),
         i < 16; i++, C += ldc ){
		C[0] = alpha*c[i] + beta*C[0]; 
	}

}	
