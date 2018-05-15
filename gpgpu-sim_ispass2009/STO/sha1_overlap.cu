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


__global__
void sha1_overlap( unsigned char *input, int chunkSize, int offset,
		   int totalThreads, int padSize, unsigned char *output ) {
  __requires(chunkSize == 52);
  __requires(offset == 4);
  __requires(totalThreads == 49152);
  __requires(padSize == 0);

  int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int chunkIndex = threadIndex * offset;
  int hashIndex  = threadIndex * SHA1_HASH_SIZE;

  if(threadIndex >= totalThreads)
    return;
  
  if ((threadIndex == (totalThreads - 1))) {
    chunkSize-= padSize;
  }

#ifdef FEATURE_SHARED_MEMORY
  
  __shared__ unsigned int sharedMemory[4 * 1024 - 32];
  
    //NOTE : SAMER : this can exceed the size of the shared memory 
  unsigned int *sharedMemoryIndex = sharedMemory + ((threadIdx.x >> 5) * 512);
  unsigned int *inputIndex = (unsigned int *)(input + chunkIndex);
  
  sha1_internal_overlap(inputIndex, sharedMemoryIndex, chunkSize, 
	       output + hashIndex );

#else
  sha1_internal(input + chunkIndex, chunkSize, output + hashIndex );
#endif /* FEATURE_SHARED_MEMORY */


}
