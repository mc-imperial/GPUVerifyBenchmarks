//pass
//--gridDim=8192 --blockDim=256

__global__ void fwtBatch2Kernel(
    float *d_Output,
    float *d_Input,
    int stride
)
{
  __requires(stride == 2048);
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int   N = blockDim.x *  gridDim.x * 4;

    float *d_Src = d_Input  + blockIdx.y * N;
    float *d_Dst = d_Output + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    float D0 = d_Src[i0];
    float D1 = d_Src[i1];
    float D2 = d_Src[i2];
    float D3 = d_Src[i3];

    float T;
    T = D0;
    D0        = D0 + D2;
    D2        = T - D2;
    T = D1;
    D1        = D1 + D3;
    D3        = T - D3;
    T = D0;
    d_Dst[i0] = D0 + D1;
    d_Dst[i1] = T - D1;
    T = D2;
    d_Dst[i2] = D2 + D3;
    d_Dst[i3] = T - D3;
}
