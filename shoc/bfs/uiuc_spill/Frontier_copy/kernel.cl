//pass
//--global_size=10240 --local_size=1024

//providence: ./BFS --algo 2 -s 2

// ****************************************************************************
// Function: Frontier_copy
//
// Purpose:
//   Copy frontier2 data to frontier
//
// Arguments:
//   frontier: array that stores the vertices to visit in the current level 
//   frontier2: alternate frontier array
//   frontier_length: length of the frontier array
//   g_mutex: mutex for implementing global barrier 
//   g_mutex2: gives the starting value of the g_mutex used in global barrier 
//   g_q_offsets: gives the offset of a block in the global queue
//   g_q_size: size of the global queue 
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
__kernel void Frontier_copy(
    __global unsigned int *frontier, 
    __global unsigned int *frontier2, 
    __global unsigned int *frontier_length,  
    __global volatile int *g_mutex, 
    __global volatile int *g_mutex2, 
    __global volatile int *g_q_offsets, 
    __global volatile int *g_q_size)
{
    unsigned int tid=get_global_id(0);

    if(tid<*frontier_length)
    {
        frontier[tid]=frontier2[tid];
    }
}
