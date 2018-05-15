//pass
//--blockDim=96 --gridDim=96

// N-queen for CUDA
//
// Copyright(c) 2008 Ping-Che Chen

#define THREAD_NUM		96

/* --------------------------------------------------------------------------
 * This is a non-recursive version of n-queen backtracking solver for CUDA.
 * It receives multiple initial conditions from a CPU iterator, and count
 * each conditions.
 * --------------------------------------------------------------------------
 */

__global__ void solve_nqueen_cuda_kernel(int n, int mark, unsigned int* total_masks, unsigned int* total_l_masks, unsigned int* total_r_masks, unsigned int* results, int total_conditions)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int idx = bid * blockDim.x + tid;

	__shared__ unsigned int mask[THREAD_NUM][10];
	__shared__ unsigned int l_mask[THREAD_NUM][10];
	__shared__ unsigned int r_mask[THREAD_NUM][10];
	__shared__ unsigned int m[THREAD_NUM][10];

	__shared__ unsigned int sum[THREAD_NUM];

	const unsigned int t_mask = (1 << n) - 1;
	int total = 0;
	int i = 0;
	unsigned int index;

	if(idx < total_conditions) {
		mask[tid][i] = total_masks[idx];
		l_mask[tid][i] = total_l_masks[idx];
		r_mask[tid][i] = total_r_masks[idx];
		m[tid][i] = mask[tid][i] | l_mask[tid][i] | r_mask[tid][i];

		while(i >= 0) {
			if((m[tid][i] & t_mask) == t_mask) {
				i--;
			}
			else {
				index = (m[tid][i] + 1) & ~m[tid][i];
				m[tid][i] |= index;
				if((index & t_mask) != 0) {
					if(i + 1 == mark) {
						total++;
						i--;
					}
					else {
						mask[tid][i + 1] = mask[tid][i] | index;
						l_mask[tid][i + 1] = (l_mask[tid][i] | index) << 1;
						r_mask[tid][i + 1] = (r_mask[tid][i] | index) >> 1;
						m[tid][i + 1] = (mask[tid][i + 1] | l_mask[tid][i + 1] | r_mask[tid][i + 1]);
						i++;
					}
				}
				else {
					i --;
				}
			}
		}

		sum[tid] = total;
	}
	else {
		sum[tid] = 0;
	}

	__syncthreads();

	// reduction
	if(tid < 64 && tid + 64 < THREAD_NUM) { sum[tid] += sum[tid + 64]; } __syncthreads();
	if(tid < 32) { sum[tid] += sum[tid + 32]; } __syncthreads();
	if(tid < 16) { sum[tid] += sum[tid + 16]; } __syncthreads();
	if(tid < 8) { sum[tid] += sum[tid + 8]; } __syncthreads();
	if(tid < 4) { sum[tid] += sum[tid + 4]; } __syncthreads();
	if(tid < 2) { sum[tid] += sum[tid + 2]; } __syncthreads();
	if(tid < 1) { sum[tid] += sum[tid + 1]; } __syncthreads();

	if(tid == 0) {
		results[bid] = sum[0];
	}
}
