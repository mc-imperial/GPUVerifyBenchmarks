//pass
//--blockDim=192 --gridDim=512

#include "md5_common.h"

/*===========================================================================

FUNCTION MD5

DESCRIPTION
  Main md5 hash function

DEPENDENCIES
  GPU must be initialized

RETURN VALUE
  output: the hash result

===========================================================================*/

__global__
void md5( unsigned char *input, int chunkSize, int totalThreads,
          int padSize, unsigned char *scratch) {
  __requires(chunkSize == 1012);
  __requires(totalThreads == 98304);
  __requires(padSize == 0);
  
  int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int chunkIndex = threadIndex * chunkSize;
  int hashIndex  = threadIndex * MD5_HASH_SIZE;

  if(threadIndex >= totalThreads)
    return;
  
  if ((threadIndex == (totalThreads - 1)) && (padSize > 0)) {
    for(int i = 0 ; i < padSize ; i++)
      input[chunkIndex + chunkSize - padSize + i] = 0;
  }


#ifdef FEATURE_SHARED_MEMORY
  
  __shared__ unsigned int sharedMemory[4 * 1024 - 32];
  
  // 512 words are allocated for every warp of 32 threads
  unsigned int *sharedMemoryIndex = sharedMemory + ((threadIdx.x >> 5) * 512);
  unsigned int *inputIndex = (unsigned int *)(input + chunkIndex);
  
  md5_internal(inputIndex, sharedMemoryIndex, chunkSize, 
	       scratch + hashIndex );

#else
  md5_internal(input + chunkIndex, chunkSize, scratch + hashIndex );
#endif /* FEATURE_SHARED_MEMORY */

}
