//pass
//--gridDim=[2,2,1]        --blockDim=[2,2,2]

#include "printf.h"

#define CUPRINTF cuPrintf

__global__ void testKernel(int val)
{
    CUPRINTF("\tValue is:%d\n", val);
}
