//xfail:NOT_ALL_VERIFIED
//--gridDim=[2,1,1]        --blockDim=[32,1,1]

#define assert __assert

__global__ void testKernel(int N)
{
    int gtid = blockIdx.x*blockDim.x + threadIdx.x ;
    assert(gtid < N) ;
}
