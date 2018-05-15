//pass
//--global_size=256 --local_size=256

//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void Fan1(__global float *m_dev,
                  __global float *a_dev,
                  __global float *b_dev,
                  const int size,
                  const int t) {
    __requires(size == 256);
    __requires(t >= 0 & t <= 254);

    int globalId = get_global_id(0);
                              
    if (globalId < size-1-t) {
         *(m_dev + size * (globalId + t + 1)+t) = *(a_dev + size * (globalId + t + 1) + t) / *(a_dev + size * t + t);    
    }
}
