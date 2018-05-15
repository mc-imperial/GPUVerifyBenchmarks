//pass
//--global_size=1024 --local_size=1024

//providence: ./BFS --algo 2 -s 2

//An Effective GPU Implementation of Breadth-First Search, Lijuan Luo,
//Martin Wong,Wen-mei Hwu ,
//Department of Electrical and Computer Engineering, 
//University of Illinois at Urbana-Champaign
// ****************************************************************************
// Function: BFS_kernel_one_block
//
// Purpose:
//   Perform BFS on the given graph when the frontier length is within one
//   thread block (i.e max number of threads per block)
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level 
//   frontier_len: length of the given frontier array 
//   visited: mask that tells if a vertex is currently in frontier
//   cost: array that stores the cost to visit each vertex 
//   edgeArray: array that gives offset of a vertex in edgeArrayAux
//   edgeArrayAux: array that gives the edge list of a vertex 
//   numVertices: number of vertices in the given graph 
//   numEdges: number of edges in the given graph 
//   frontier_length: length of the new frontier array
//   max_local_mem: max size of the shared memory queue
//   b_q: block level queue
//   b_q2: alterante block level queue
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void BFS_kernel_one_block(

    volatile __global unsigned int *frontier,
    unsigned int frontier_len,
    volatile __global int *visited,
    volatile __global unsigned int *cost,
    __global unsigned int *edgeArray,
    __global unsigned int *edgeArrayAux,
    unsigned int numVertices,
    unsigned int numEdges,
    volatile __global unsigned int *frontier_length,
    const unsigned int max_local_mem,

    //the block queues of size MAX_THREADS_PER_BLOCK
    volatile __local unsigned int *b_q,
    volatile __local unsigned int *b_q2)
{
    __requires(frontier_len == 1);
    // numVertices irrelevant
    // numEdges irrelevant
    __requires(max_local_mem == 1024);

    volatile __local unsigned int b_offset[1];
    volatile __local unsigned int b_q_length[1];

    //get the threadId
    unsigned int tid = get_local_id(0);
    //copy frontier queue from global queue to local block queue
    if(tid<frontier_len)
    {
        b_q[tid]=frontier[tid];
    }

    unsigned int f_len=frontier_len;
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    while(1)
    {
        //Initialize the block queue size to 0
        if(tid==0)
        {
            b_q_length[0]=0;
            b_offset[0]=0;
        }
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        if(tid<f_len)
        {
            //get the nodes to traverse from block queue
            unsigned int node_to_process=b_q[tid];
            
            visited[node_to_process]=0;
            //get the offsets of the vertex in the edge list
            unsigned int offset = edgeArray[node_to_process];
            unsigned int next   = edgeArray[node_to_process+1];

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
                            //increment the local queue size
                            unsigned int t=atomic_add(&b_q_length[0],1);
                            if(t< max_local_mem)
                            {
                                b_q2[t]=nid;
                            }
                            //write to global memory if shared memory full
                            else
                            {
                                int off=atomic_add(&b_offset[0],1);
                                frontier[off]=nid;
                            }
                        }
                }
                offset++;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        //copy block queue from b_q2 to b_q
        if(tid<max_local_mem)
            b_q[tid]=b_q2[tid];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        //if traversal complete exit
        if(b_q_length[0]==0)
        { 
            if(tid==0)
                frontier_length[0]=0;

            return;
        }
        // if frontier exceeds one block in size copy block queue to
        //global queue and exit
        else if( b_q_length[0] > get_local_size(0) || 
                 b_q_length[0] > max_local_mem)
        {
            if(tid<(b_q_length[0]-b_offset[0]))
                frontier[b_offset[0]+tid]=b_q[tid];
            if(tid==0)
            {
                frontier_length[0] = b_q_length[0];
            }
            return; 
        }
        f_len=b_q_length[0];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
}
