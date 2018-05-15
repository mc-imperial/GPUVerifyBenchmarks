//pass
//--local_size=[16,1] --num_groups=[127,1]

// Also uses num_groups=[126,1] .. [1,1]

#include "../common.h"

__kernel void 
nw_kernel2(__global int  * reference_d, 
		   __global int  * input_itemsets_d, 
		   __global int  * output_itemsets_d, 
		   __local	int  * input_itemsets_l,
		   __local	int  * reference_l,
           int cols,
           int penalty,
           int blk,
           int block_width,
           int worksize,
           int offset_r,
           int offset_c
    )
{  
  __requires(cols == 2049);
  __requires(penalty == 10);
  __requires(blk == 127);
  __requires(block_width == 128);
  __requires(worksize == 2048);
  __requires(offset_r == 0);
  __requires(offset_c == 0);

	int bx = get_group_id(0);	
	//int bx = get_global_id(0)/BLOCK_SIZE;
   
    // Thread index
    int tx = get_local_id(0);
    
    // Base elements
    int base = offset_r * cols + offset_c;
    
    int b_index_x = bx + block_width - blk  ;
	int b_index_y = block_width - bx -1;
	
	
	int index   =   base + cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
	int index_n   = base + cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
	int index_w   = base + cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
	int index_nw =  base + cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
    
	if (tx == 0)
		SCORE(tx, 0) = input_itemsets_d[index_nw];

	for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
		REF(ty, tx) =  reference_d[index + cols * ty];

	barrier(CLK_LOCAL_MEM_FENCE);

	SCORE((tx + 1), 0) = input_itemsets_d[index_w + cols * tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	SCORE(0, (tx + 1)) = input_itemsets_d[index_n];
  
	barrier(CLK_LOCAL_MEM_FENCE);
  
	for( int m = 0 ; m < BLOCK_SIZE ; m++){
	
	  if ( tx <= m ){
	  
		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

         SCORE(t_index_y, t_index_x) = maximum(  SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
		                                         SCORE((t_index_y),   (t_index_x-1)) - (penalty), 
		 										 SCORE((t_index_y-1), (t_index_x))   - (penalty));
	  }
	  barrier(CLK_LOCAL_MEM_FENCE);
    }

	for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){
 
		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
		                                         SCORE((t_index_y),   (t_index_x-1)) - (penalty), 
		 										 SCORE((t_index_y-1), (t_index_x))   - (penalty));
	   
	  }

	  barrier(CLK_LOCAL_MEM_FENCE);
	}

	for ( int ty = 0 ;
          __global_invariant(__write_implies(input_itemsets_d, (__write_offset_bytes(input_itemsets_d)/sizeof(int) - index)%cols == 0)),
          __global_invariant(__write_implies(input_itemsets_d, (__write_offset_bytes(input_itemsets_d)/sizeof(int) - index)/cols < BLOCK_SIZE)),
          ty < BLOCK_SIZE ; ty++)
		input_itemsets_d[index + ty * cols] = SCORE((ty+1), (tx+1));
	
    
    return;
  
}
