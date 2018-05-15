//pass
//--num_groups=[16,32] --local_size=[16,8]

__kernel void readInCache(int n, __global float *d_out,      
    __read_only image2d_t img, sampler_t samp) 
{
    int2 ridx = (int2)(get_global_id(0),get_global_id(1));   
    int idx = ridx.x*get_global_size(1) + ridx.y;
    float sum = 0.0f;                                        
    for (int i = 0; i < n; i++)                              
    {                                                        
        float4 x = read_imagef(img, samp, ridx); 
        sum += x.x;                                          
    }                                                        
    d_out[idx] = sum;                                        
}                                                            
