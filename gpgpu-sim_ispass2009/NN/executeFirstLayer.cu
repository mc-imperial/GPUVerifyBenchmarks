//pass
//--gridDim=[6,10] --blockDim=[13,13]

__constant__ int kernelTemplate[25] = {
        0,  1,  2,  3,  4,
        29, 30, 31, 32, 33,
        58, 59, 60, 61, 62,
        87, 88, 89, 90, 91,
        116,117,118,119,120 };

__global__ void executeFirstLayer(float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU)
{
	int blockID=blockIdx.x;
	int pixelX=threadIdx.x;
	int pixelY=threadIdx.y;


	int weightBegin=blockID*26;
	int windowX=pixelX*2;
	int windowY=pixelY*2;

	float result=0;

	result+=Layer1_Weights_GPU[weightBegin];

	++weightBegin;

//for(int i=0;i<25;++i)
//{
//	result+=Layer1_Neurons_GPU[(windowY*29+windowX+kernelTemplate[i])+(29*29*blockIdx.y)]*Layer1_Weights_GPU[weightBegin+i];
//}

//result=(1.7159*tanhf(0.66666667*result));

	Layer2_Neurons_GPU[(13*13*blockID+pixelY*13+pixelX)+(13*13*6*blockIdx.y)]=result;

}
