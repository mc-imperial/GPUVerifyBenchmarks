//pass
//--global_size=4096 --local_size=1024

//providence: ./BFS --algo 2 -s 2 --global-barrier

//S. Xiao and W. Feng, .Inter-block GPU communication via fast barrier 
//synchronization,.Technical Report TR-09-19, 
//Dept. of Computer Science, Virginia Tech
// ****************************************************************************
// Function: __gpu_sync
//
// Purpose:
//   Implements global barrier synchronization across thread blocks. Thread 
//   blocks must be limited to number of multiprocessors available
//
// Arguments:
//   blocks_to_synch: the number of blocks across which to implement the barrier
//   g_mutex: keeps track of number of blocks that are at barrier
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
static __attribute__((always_inline))
void __gpu_sync(int blocks_to_synch , volatile __global unsigned int *g_mutex)
{
    //thread ID in a block
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    int tid_in_block= get_local_id(0);
    

    // only thread 0 is used for synchronization
    if (tid_in_block == 0) 
    {
        atomic_add(g_mutex, 1);               
        //only when all blocks add 1 to g_mutex will
        //g_mutex equal to blocks_to_synch
        while(g_mutex[0] < blocks_to_synch)
        {
        }

    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}

// ****************************************************************************
// Function: BFS_kernel_SM_block
//
// Purpose:
//   Perform BFS on the given graph when the frontier length is greater than 
//   one thread block but less than number of Streaming Multiprocessor(SM) 
//   thread blocks (i.e max threads per block * SM blocks)
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level 
//   frontier_len: length of the given frontier array 
//   frontier2: alternate frontier array
//   visited: mask that tells if a vertex is currently in frontier
//   cost: array that stores the cost to visit each vertex 
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex 
//   numVertices: number of vertices in the given graph 
//   numEdges: number of edges in the given graph 
//   frontier_length: length of the new frontier array
//   g_mutex: mutex for implementing global barrier 
//   g_mutex2: gives the starting value of the g_mutex used in global barrier 
//   g_q_offsets: gives the offset of a block in the global queue
//   g_q_size: keeps track of the size of frontier in intermediate iterations
//   max_local_mem: max size of the shared memory queue
//   b_q: block level queue
//
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void BFS_kernel_SM_block(

    volatile __global unsigned int *frontier,
    unsigned int frontier_len,
    volatile __global unsigned int *frontier2,
    volatile __global int *visited, 
    volatile __global unsigned int *cost,
    __global unsigned int *edgeArray, 
    __global unsigned int *edgeArrayAux,
    unsigned int numVertices, 
    unsigned int numEdges, 
    volatile __global unsigned int *frontier_length,
    volatile __global unsigned int *g_mutex, 
    volatile __global unsigned int *g_mutex2, 
    volatile __global unsigned int *g_q_offsets, 
    volatile __global unsigned int *g_q_size,
    const unsigned int max_local_mem,

    //block queue
    volatile __local unsigned int *b_q)
{
    __requires(frontier_len == 2048);
    //numVertices irrelevant
    //numEdges irrelevant
    __requires(max_local_mem == 1024);

    volatile __local unsigned int b_q_length[1];
    volatile __local unsigned int b_offset[1];
    //get the threadId
    unsigned int tid=get_global_id(0);
    unsigned int lid=get_local_id(0);

    int loop_index=0;
    unsigned int l_mutex=g_mutex2[0];
    unsigned int f_len=frontier_len;
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    while(1)
    {
        //Initialize the block queue size to 0
        if (lid==0)
        {
            b_q_length[0]=0;
            b_offset[0]=0;
        }
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        if(tid<f_len)
        {
            unsigned int node_to_process;  
            
            //get the node to traverse from block queue
            if(loop_index==0)
               node_to_process=frontier[tid];
            else
               node_to_process=frontier2[tid]; 

            //node removed from frontier
            visited[node_to_process]=0;
            //get the offsets of the vertex in the edge list
            unsigned int offset=edgeArray[node_to_process];
            unsigned int next=edgeArray[node_to_process+1];

            //Iterate through the neighbors of the vertex
            while(offset<next)
            {
                //get neighbor
                unsigned int nid=edgeArrayAux[offset];
                //get its cost
                unsigned int v=atomic_min(&cost[nid],cost[node_to_process]+1);
                //if cost is less than previously set add to frontier
                if(v>cost[node_to_process]+1)
                {
                    int is_in_frontier=atomic_xchg(&visited[nid],1);
                    //if node already in frontier do nothing
                    if(is_in_frontier==0)
                    {
                        //increment the warp queue size
                        unsigned int t=atomic_add(&b_q_length[0],1);
                        if(t<max_local_mem)
                        {
                            b_q[t]=nid;
                        }
                        //write to global memory if shared memory full
                        else
                        {
                            int off=atomic_add(g_q_offsets,1);
                            if(loop_index==0)
                                frontier2[off]=nid;
                            else
                                frontier[off]=nid;
                        } 
                    }
                }
                offset++;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        //get block queue offset in global queue
        if(lid==0)
        {
            if(b_q_length[0] > max_local_mem)
            {
                b_q_length[0] = max_local_mem;
            }
            b_offset[0]=atomic_add(g_q_offsets,b_q_length[0]);
        }

        //global barrier
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
		l_mutex+=get_num_groups(0);
		__gpu_sync(l_mutex,g_mutex);

        //store frontier size
        if(tid==0)
        {
            g_q_size[0]=g_q_offsets[0];
            g_q_offsets[0]=0;
        }

        //copy block queue to global queue
        if(lid < b_q_length[0])
        {
            if(loop_index==0)
                frontier2[lid+b_offset[0]]=b_q[lid];
            else
                frontier[lid+b_offset[0]]=b_q[lid];
        }
        
        //global barrier
		l_mutex+=get_num_groups(0);
		__gpu_sync(l_mutex,g_mutex);

        //exit if frontier size exceeds SM blocks or is less than 1 block
        if(g_q_size[0] < get_local_size(0) ||
            g_q_size[0] > get_local_size(0) * get_num_groups(0))
                break;                                                  

        loop_index=(loop_index+1)%2;
        //store the current frontier size
        f_len=g_q_size[0];
    }

    if(loop_index==0)
    {
        for(int i=tid;i<g_q_size[0];i += get_global_size(0))
               frontier[i]=frontier2[i];
    }
    if(tid==0)
    {
        frontier_length[0]=g_q_size[0];
    }
}
