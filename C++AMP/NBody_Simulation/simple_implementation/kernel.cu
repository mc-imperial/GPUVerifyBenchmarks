//pass
//--blockDim=256 --gridDim=1024

#include <cuda.h>

#define TILE_SIZE							256
#define SOFTENING_SQUARED 					0.0000015625f
#define _FG									(6.67300e-11f*10000.0f)
#define F_PARTICLE_MASS						(_FG*10000.0f*10000.0f)
#define DELTA_TIME 							0.1f
#define DAMPENING 							1.0f

#define UINT unsigned int


// GPU based functions
static __attribute__((always_inline))
__device__ void bodybody_interaction(float4 *acc, const float4 my_curr_pos, float4 other_element_old_pos)
{
    float4 r = other_element_old_pos - my_curr_pos;
    
    float dist_sqr = r.x*r.x + r.y*r.y + r.z*r.z;
    dist_sqr += SOFTENING_SQUARED;
    
    float inv_dist = rsqrt(dist_sqr);
    float inv_dist_cube =  inv_dist*inv_dist*inv_dist;
    
    float s = F_PARTICLE_MASS*inv_dist_cube;
	
    (*acc) += r*s;
}

__global__ void simple_implementation(float4* data_in_pos, float4* data_in_vel, float4* data_out_pos, float4* data_out_vel, unsigned int num_bodies)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
	{
        float4 p_pos;
        float4 p_vel;

        p_pos = data_in_pos[idx];
        p_vel = data_in_vel[idx];
      //float4 acc = (float4)(0, 0, 0, 0);
        float4 acc;
        acc.x = 0; acc.y = 0; acc.z = 0; acc.w = 0;

        // Update current particle using all other particles
        for (UINT j = 0; j < num_bodies; j++) 
        {
	        bodybody_interaction(&acc, p_pos, data_in_pos[j]);
        }

        p_vel += acc*DELTA_TIME;
        p_vel *= DAMPENING;

        p_pos += p_vel*DELTA_TIME;

        data_out_pos[idx] = p_pos;
        data_out_vel[idx] = p_vel;
#ifdef MUTATION
        data_out_vel[idx+1] = data_out_vel[idx+1];
         /* BUGINJECT: ADD_ACCESS, UP */
#endif
	}
}
