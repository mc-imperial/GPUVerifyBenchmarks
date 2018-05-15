//pass
//--gridDim=[10,10] --blockDim=1

__global__ void executeFourthLayer(float *Layer4_Neurons_GPU,float *Layer4_Weights_GPU,float *Layer5_Neurons_GPU)
{
	int blockID=blockIdx.x;
	//int pixelY=threadIdx.y;


	int weightBegin=blockID*101;
 
	float result=0;

	result+=Layer4_Weights_GPU[weightBegin];

	++weightBegin;

    for (int i=0; i<100; ++i )
    {
		result+=Layer4_Neurons_GPU[i+(100*blockIdx.y)]*Layer4_Weights_GPU[weightBegin+i];
    }

	result=(1.7159*tanhf(0.66666667*result));

	Layer5_Neurons_GPU[blockID+(10*blockIdx.y)]=result;
}
