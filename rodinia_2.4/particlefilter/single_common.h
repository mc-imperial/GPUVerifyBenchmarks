#ifndef SINGLE_COMMON_H
#define SINGLE_COMMON_H

static __attribute__((always_inline))
float d_randu(__global int * seed, int index)
{

	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index] / ((float) M));
}


#endif

