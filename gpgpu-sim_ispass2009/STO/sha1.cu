//pass
//--blockDim=32 --gridDim=2

#include "sha1_common.h"

/*==========================================================================
                                SHA1 KERNEL

* Copyright (c) 2008, NetSysLab at the University of British Columbia
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the University nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY NetSysLab ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL NetSysLab BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DESCRIPTION
  CPU version of the storeGPU library.


==========================================================================*/

/*===========================================================================

FUNCTION SHA1

DESCRIPTION
  Main sha1 hash function

DEPENDENCIES
  GPU must be initialized

RETURN VALUE
  output: the hash result

===========================================================================*/
__global__
void sha1( unsigned char *input, int chunkSize, int totalThreads,
	   int padSize, unsigned char *scratch ) {
  __requires(chunkSize == 1012);
  __requires(totalThreads == 98304);
  __requires(padSize == 0);
  
  // get the current thread index
  int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int chunkIndex = threadIndex * chunkSize;
  int hashIndex  = threadIndex * SHA1_HASH_SIZE;

  if(threadIndex >= totalThreads)
    return;
  
  if ((threadIndex == (totalThreads - 1)) && (padSize > 0)) {
    for(int i = 0 ; i < padSize ; i++)
      input[chunkIndex + chunkSize - padSize + i] = 0;	
  }
  
#ifdef FEATURE_SHARED_MEMORY
  
  __shared__ unsigned int sharedMemory[4 * 1024 - 32];
  
  unsigned int *sharedMemoryIndex = sharedMemory + ((threadIdx.x >> 5) * 512);
  unsigned char *tempInput = input + chunkIndex;
  unsigned int *inputIndex = (unsigned int *)(tempInput);
  
  sha1_internal(inputIndex, sharedMemoryIndex, chunkSize, 
	       scratch + hashIndex );

#else
  sha1_internal(input + chunkIndex, chunkSize, scratch + hashIndex );
#endif /* FEATURE_SHARED_MEMORY */

}
