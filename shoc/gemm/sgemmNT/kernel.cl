//pass
//--global_size=[64,64] --local_size=[16,4]

#include "../common.h"

__kernel void sgemmNT( __global const FPTYPE *A, int lda,
                       __global const FPTYPE *B, int ldb,
                       __global FPTYPE *C, int ldc, int k,
                       FPTYPE alpha, FPTYPE beta )
{
  __requires(ldc == 256);

	const int inx = get_local_id(0);
	const int iny = get_local_id(1);
	const int ibx = get_group_id(0) * 64;
	const int iby = get_group_id(1) * 16;
	const int id  = inx + iny*16;

        int i, counter = 0;

	A += ibx + id;
	B += iby + inx + (iny*ldb);
	C += ibx + id  + (iby*ldc );
	
	FPTYPE a[4];
	for(i=0; i<4; ++i){ a[i] = A[i*lda]; }
	__private FPTYPE b;
	b = B[0];

	A += 4*lda;
	B += 4*ldb;
        counter+= 4*ldb;
    
	__local FPTYPE bs[4][16];
	FPTYPE c[16];
        for(i=0; i<16; ++i){
            c[i] = 0.0;
        }
    
	do
	{
	        __private FPTYPE as[4];
		for(i=0; i<4; ++i){ as[i] = a[i]; }
		
		bs[iny][inx] = b;
  		barrier(CLK_LOCAL_MEM_FENCE);
		
		a[0] = A[0*lda];
		a[1] = A[1*lda];
		a[2] = A[2*lda];
		a[3] = A[3*lda];
		b    = B[0];
		
		SAXPY( as[0], bs[0], c );
		SAXPY( as[1], bs[1], c );
		SAXPY( as[2], bs[2], c );
		SAXPY( as[3], bs[3], c );

		A += 4*lda;
		B += 4*ldb;
                counter += 4*ldb;
  		barrier(CLK_LOCAL_MEM_FENCE);
		
	} while( counter < k*ldb );
	
	bs[iny][inx] = b;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	SAXPY( a[0], bs[0], c );
	SAXPY( a[1], bs[1], c );
	SAXPY( a[2], bs[2], c );
	SAXPY( a[3], bs[3], c );

    for( int i = 0;
         __global_invariant((__ptr_offset_bytes(C)/sizeof(FPTYPE) - (ibx + id + iby * ldc)) == ldc * i),
         __global_invariant(__write_implies(C, (__write_offset_bytes(C)/sizeof(FPTYPE) - (ibx + id + iby * ldc))/ldc < 16)),
         __global_invariant(__read_implies(C, (__read_offset_bytes(C)/sizeof(FPTYPE) - (ibx + id + iby * ldc))/ldc < 16)),
        i < 16; i++, C += ldc ){
		C[0] = alpha*c[i] + beta*C[0];
        }
}
