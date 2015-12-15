#ifndef _REG_OPTIMISER_KERNELS_CU
#define _REG_OPTIMISER_KERNELS_CU

__device__ __constant__ int c_NodeNumber;
__device__ __constant__ float c_ScalingFactor;

texture<float4, 1, cudaReadModeElementType> gradientImageTexture;
texture<float4, 1, cudaReadModeElementType> conjugateGTexture;
texture<float4, 1, cudaReadModeElementType> conjugateHTexture;
texture<float4, 1, cudaReadModeElementType> controlPointTexture;

__global__ void reg_initialiseConjugateGradient_kernel(	float4 *conjugateG_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_NodeNumber){
        float4 gradValue = tex1Dfetch(gradientImageTexture,tid);
        conjugateG_d[tid] = make_float4(-gradValue.x, -gradValue.y, -gradValue.z,0.0f);
    }
}


__global__ void reg_GetConjugateGradient1_kernel(float2 *sum)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_NodeNumber){
        float4 valueH = tex1Dfetch(conjugateHTexture,tid);
        float4 valueG = tex1Dfetch(conjugateGTexture,tid);
        float gg= valueG.x*valueH.x + valueG.y*valueH.y + valueG.z*valueH.z;

        float4 grad = tex1Dfetch(gradientImageTexture,tid);
        float dgg= (grad.x+valueG.x)*grad.x + (grad.y+valueG.y)*grad.y + (grad.z+valueG.z)*grad.z;

        sum[tid]=make_float2(dgg,gg);
    }
}

__global__ void reg_GetConjugateGradient2_kernel(	float4 *nodeNMIGradientArray_d,
                                                        float4 *conjugateG_d,
                                                        float4 *conjugateH_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_NodeNumber){
        // G = - grad
        float4 gradGValue = nodeNMIGradientArray_d[tid];
        gradGValue = make_float4(-gradGValue.x, -gradGValue.y, -gradGValue.z, 0.0f);
        conjugateG_d[tid]=gradGValue;

        // H = G + gam * H
        float4 gradHValue = conjugateH_d[tid];
        gradHValue=make_float4(
                gradGValue.x + c_ScalingFactor * gradHValue.x,
                gradGValue.y + c_ScalingFactor * gradHValue.y,
                gradGValue.z + c_ScalingFactor * gradHValue.z,
                0.0f);
        conjugateH_d[tid]=gradHValue;
        nodeNMIGradientArray_d[tid]=make_float4(-gradHValue.x, -gradHValue.y, -gradHValue.z, 0.0f);
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
__global__ void reg_getEuclideanDistance_kernel(float *distance_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_NodeNumber){

        float4 gradValue = tex1Dfetch(gradientImageTexture,tid);
        distance_d[tid] = sqrtf(gradValue.x*gradValue.x + gradValue.y*gradValue.y + gradValue.z*gradValue.z);
    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
__global__ void reg_updateControlPointPosition_kernel(float4 *controlPointImageArray_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_NodeNumber){
		float scaling = c_ScalingFactor;
		float4 value = tex1Dfetch(controlPointTexture,tid);
		float4 gradValue = tex1Dfetch(gradientImageTexture,tid);
		value.x += scaling * gradValue.x;
		value.y += scaling * gradValue.y;
		value.z += scaling * gradValue.z;
		value.w = 0.0f;
		controlPointImageArray_d[tid]=value;

    }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif // _REG_OPTIMISER_KERNELS_CU
