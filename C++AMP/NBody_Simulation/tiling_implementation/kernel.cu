//pass
//--blockDim=256 --gridDim=1024

#include <cuda.h>

#define TILE_SIZE							256
#define SOFTENING_SQUARED 					0.0000015625f
#define _FG									(6.67300e-11f*10000.0f)
#define F_PARTICLE_MASS						(_FG*10000.0f*10000.0f)
#define DELTA_TIME 							0.1f
#define DAMPENING 							1.0f


#define to_d3dxv4(X) X
#define to_float4(X) X

#define D3DXVECTOR4 float4

#define UINT unsigned int

// GPU based functions
static __attribute__((always_inline))
__device__ void bodybody_interaction(float4* acc, const float4 my_curr_pos, float4 other_element_old_pos)
{
    float4 r = other_element_old_pos - my_curr_pos;
    
    float dist_sqr = r.x*r.x + r.y*r.y + r.z*r.z;
    dist_sqr += SOFTENING_SQUARED;
    
    float inv_dist = rsqrt(dist_sqr);
    float inv_dist_cube =  inv_dist*inv_dist*inv_dist;
    
    float s = F_PARTICLE_MASS*inv_dist_cube;
	
    (*acc) += r*s;
}

__global__ void tiling_implementation(float4* data_in_pos, float4* data_in_vel, float4* data_out_pos, float4* data_out_vel, int offset, int size, int num_bodies)
{
    UINT num_of_tiles = num_bodies/TILE_SIZE;

	{
		__shared__ D3DXVECTOR4 tile_mem[TILE_SIZE];
    
		int idx_local = threadIdx.x;
		int idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    
		idx_global += offset;

		float4 p_pos;
    float4 p_vel;
		p_pos = data_in_pos[idx_global];
		p_vel = data_in_vel[idx_global];
		//float4 acc = (float4)(0, 0, 0, 0);
		float4 acc;
    acc.x = 0; acc.y = 0; acc.z = 0; acc.w = 0;
    
		// Update current particle using all other particles
		int particle_idx = idx_local;
		for (UINT tile = 0;
                                tile <num_of_tiles; tile++)
		{
			// Cache a tile of particles into shared memory to increase IO efficiency
			tile_mem[idx_local] = to_d3dxv4(data_in_pos[particle_idx]);

#ifndef MUTATION
       /* BUGINJECT: REMOVE_BARRIER, DOWN */
      __syncthreads();
#endif
        
      // Unroll size should be multile of TILE_SIZE
			// Unrolling 4 helps improve perf on both ATI and nVidia cards
			// 4 is the sweet spot - increasing further adds no perf improvement while decreasing reduces perf
			for (UINT j = 0; j < TILE_SIZE; j+=4 )
			{
				bodybody_interaction(&acc, p_pos, to_float4(tile_mem[j+0]));
				bodybody_interaction(&acc, p_pos, to_float4(tile_mem[j+1]));
				bodybody_interaction(&acc, p_pos, to_float4(tile_mem[j+2]));
				bodybody_interaction(&acc, p_pos, to_float4(tile_mem[j+3]));
			}
      __syncthreads();
      particle_idx += TILE_SIZE;
		}

    p_vel += acc*DELTA_TIME;
		p_vel *= DAMPENING;

		p_pos += p_vel*DELTA_TIME;

		data_out_pos[idx_global] = p_pos;
		data_out_vel[idx_global] = p_vel;
	}

}
