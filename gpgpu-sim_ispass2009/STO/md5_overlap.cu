//pass
//--blockDim=32 --gridDim=2

#include "md5_common.h"

__global__
void md5_overlap( unsigned char *input, int chunkSize, int offset,
		  int totalThreads, int padSize, unsigned char *output ) {
  __requires(chunkSize == 52);
  __requires(offset == 4);
  __requires(totalThreads == 49152);
  __requires(padSize == 0);

  int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int chunkIndex = threadIndex * offset;
  int hashIndex  = threadIndex * MD5_HASH_SIZE;


  if(threadIndex >= totalThreads)
    return;
  
  if ((threadIndex == (totalThreads - 1))) {
    chunkSize-= padSize;
  }


#ifdef FEATURE_SHARED_MEMORY
  
  __shared__ unsigned int sharedMemory[4 * 1024 - 32];
  
  unsigned int *sharedMemoryIndex = sharedMemory + ((threadIdx.x >> 5) * 512);
  unsigned int *inputIndex = (unsigned int *)(input + chunkIndex);
  
  md5_internal_overlap(inputIndex, sharedMemoryIndex, chunkSize, 
	       output + hashIndex );

#else
  md5_internal(input + chunkIndex, chunkSize, output + hashIndex );
#endif /* FEATURE_SHARED_MEMORY */


}
