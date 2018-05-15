//pass
//--gridDim=[100,10] --blockDim=1

__global__ void executeThirdLayer(float *Layer3_Neurons_GPU, float *Layer3_Weights_GPU,float *Layer4_Neurons_GPU)
{
	int blockID=blockIdx.x;
	//int pixelY=threadIdx.y;


	int weightBegin=blockID*1251;
 
	float result=0;

	result+=Layer3_Weights_GPU[weightBegin];

	++weightBegin;

    for (int i=0; i<1250; ++i )
    {
		result+=Layer3_Neurons_GPU[i+(1250*blockIdx.y)]*Layer3_Weights_GPU[weightBegin+i];
    }

	result=(1.7159*tanhf(0.66666667*result));

	Layer4_Neurons_GPU[blockID+(100*blockIdx.y)]=result;

}
