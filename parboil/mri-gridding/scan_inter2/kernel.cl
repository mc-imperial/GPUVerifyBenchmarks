//pass
//--local_size=[128] --num_groups=[256]

// Also run as --local_size=[128] --num_groups=[1] and --local_size=[64] --num_groups=[1]

#define DYN_LOCAL_MEM_SIZE 544

#define BLOCK_SIZE 1024
#define GRID_SIZE 65535
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

//#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#define LNB LOG_NUM_BANKS
#define CONFLICT_FREE_OFFSET(index) (((unsigned int)(index) >> min((unsigned int)(LNB)+(index), (unsigned int)(32-(2*LNB))))>>(2*LNB))

__kernel void scan_inter2_kernel(__global unsigned int* data, unsigned int iter)
{
  __requires(iter == 1);
    
    __local unsigned int s_data[DYN_LOCAL_MEM_SIZE];

    unsigned int thid = get_local_id(0);
    unsigned int gthid = get_global_id(0);
    unsigned int gi = 2*iter*gthid;
    unsigned int g_ai = gi + iter - 1;
    unsigned int g_bi = g_ai + iter;

    unsigned int s_ai = 2*thid;
    unsigned int s_bi = 2*thid + 1;

    s_ai += CONFLICT_FREE_OFFSET(s_ai);
    s_bi += CONFLICT_FREE_OFFSET(s_bi);

    s_data[s_ai] = data[g_ai];
    s_data[s_bi] = data[g_bi];

    unsigned int stride = get_local_size(0)*2;

    for (unsigned int d = 1; d <= get_local_size(0); d *= 2) {
      stride >>= 1;

      barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

      if (thid < d) {
        unsigned int i  = 2*stride*thid;
        unsigned int ai = i + stride - 1;
        unsigned int bi = ai + stride;

        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        unsigned int t  = s_data[ai];
        s_data[ai] = s_data[bi];
        s_data[bi] += t;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    data[g_ai] = s_data[s_ai];
    data[g_bi] = s_data[s_bi];
}
