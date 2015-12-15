/*
 *  _reg_tools_kernels.cu
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_TOOLS_KERNELS_CU
#define _REG_TOOLS_KERNELS_CU
/* *************************************************************** */
__device__ __constant__ int c_NodeNumber;
__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ int3 c_TargetImageDim;
__device__ __constant__ float3 c_VoxelNodeRatio;
__device__ __constant__ int3 c_ControlPointImageDim;
__device__ __constant__ int3 c_ImageDim;
__device__ __constant__ float c_Weight;
/* *************************************************************** */
texture<float4, 1, cudaReadModeElementType> controlPointTexture;
texture<float4, 1, cudaReadModeElementType> gradientImageTexture;
texture<float4, 1, cudaReadModeElementType> matrixTexture;
texture<float, 1, cudaReadModeElementType> convolutionKernelTexture;
/* *************************************************************** */
__global__ void reg_voxelCentric2NodeCentric_kernel(float4 *nodeNMIGradientArray_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_NodeNumber){

        const int3 gridSize = c_ControlPointImageDim;
        int tempIndex=tid;
        const short z =(int)(tempIndex/(gridSize.x*gridSize.y));
        tempIndex -= z*(gridSize.x)*(gridSize.y);
        const short y =(int)(tempIndex/(gridSize.x));
        const short x = tempIndex - y*(gridSize.x);

        const float3 ratio = c_VoxelNodeRatio;
        const short X = round((x-1)*ratio.x);
		const short Y = round((y-1)*ratio.y);
		const short Z = round((z-1)*ratio.z);

		const int3 imageSize = c_TargetImageDim;

		if(-1<X && X<imageSize.x && -1<Y && Y<imageSize.y && -1<Z && Z<imageSize.z){
			int index = (Z*imageSize.y+Y)*imageSize.x+X;
			float4 gradientValue = tex1Dfetch(gradientImageTexture,index);
			nodeNMIGradientArray_d[tid] = make_float4(c_Weight*gradientValue.x,
													  c_Weight*gradientValue.y,
													  c_Weight*gradientValue.z,
													  0.0f);
		}
		else nodeNMIGradientArray_d[tid]=make_float4(0, 0.0f, 0.0f, 0.0f);
    }
}
/* *************************************************************** */
__global__ void _reg_convertNMIGradientFromVoxelToRealSpace_kernel(float4 *gradient)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_NodeNumber){
        float4 voxelGradient = gradient[tid];
        float4 realGradient;
        float4 matrix = tex1Dfetch(matrixTexture,0);
        realGradient.x =	matrix.x*voxelGradient.x + matrix.y*voxelGradient.y  +
                                matrix.z*voxelGradient.z;
        matrix = tex1Dfetch(matrixTexture,1);
        realGradient.y =	matrix.x*voxelGradient.x + matrix.y*voxelGradient.y  +
                                matrix.z*voxelGradient.z;
        matrix = tex1Dfetch(matrixTexture,2);
        realGradient.z =	matrix.x*voxelGradient.x + matrix.y*voxelGradient.y  +
                                matrix.z*voxelGradient.z;

        gradient[tid]=realGradient;
    }
}

__global__ void _reg_ApplyConvolutionWindowAlongX_kernel(   float4 *smoothedImage,
                                                            int windowSize)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){

        int3 imageSize = c_ImageDim;

        int temp=tid;
        const short z=(int)(temp/(imageSize.x*imageSize.y));
        temp -= z*imageSize.x*imageSize.y;
        const short y =(int)(temp/(imageSize.x));
        short x = temp - y*(imageSize.x);

        int radius = (windowSize-1)/2;
        int index = tid - radius;
        x -= radius;

        float4 finalValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Kahan summation used here
        float3 c=make_float3(0.f,0.f,0.f), Y, t;
        float windowValue;
        for(int i=0; i<windowSize; i++){
            if(-1<x && x<imageSize.x){
                float4 gradientValue = tex1Dfetch(gradientImageTexture,index);
                windowValue = tex1Dfetch(convolutionKernelTexture,i);

                Y.x = gradientValue.x * windowValue - c.x;
                Y.y = gradientValue.y * windowValue - c.y;
                Y.z = gradientValue.z * windowValue - c.z;
                t.x = finalValue.x + Y.x;
                t.y = finalValue.y + Y.y;
                t.z = finalValue.z + Y.z;
                c.x = (t.x - finalValue.x) - Y.x;
                c.y = (t.y - finalValue.y) - Y.y;
                c.z = (t.z - finalValue.z) - Y.z;
                finalValue = make_float4(t.x, t.y, t.z, 0.f);
            }
            index++;
            x++;
        }
        smoothedImage[tid] = finalValue;
    }
    return;
}

__global__ void _reg_ApplyConvolutionWindowAlongY_kernel(float4 *smoothedImage,
                                                         int windowSize)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        int3 imageSize = c_ImageDim;

        const short z=(int)(tid/(imageSize.x*imageSize.y));
        int index = tid - z*imageSize.x*imageSize.y;
        short y=(int)(index/imageSize.x);

        int radius = (windowSize-1)/2;
        index = tid - imageSize.x*radius;
        y -= radius;

        float4 finalValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Kahan summation used here
        float3 c=make_float3(0.f,0.f,0.f), Y, t;
        float windowValue;
        for(int i=0; i<windowSize; i++){
            if(-1<y && y<imageSize.y){
                float4 gradientValue = tex1Dfetch(gradientImageTexture,index);
                windowValue = tex1Dfetch(convolutionKernelTexture,i);

                Y.x = gradientValue.x * windowValue - c.x;
                Y.y = gradientValue.y * windowValue - c.y;
                Y.z = gradientValue.z * windowValue - c.z;
                t.x = finalValue.x + Y.x;
                t.y = finalValue.y + Y.y;
                t.z = finalValue.z + Y.z;
                c.x = (t.x - finalValue.x) - Y.x;
                c.y = (t.y - finalValue.y) - Y.y;
                c.z = (t.z - finalValue.z) - Y.z;
                finalValue = make_float4(t.x, t.y, t.z, 0.f);
            }
            index += imageSize.x;
            y++;
        }
        smoothedImage[tid] = finalValue;
    }
    return;
}


__global__ void _reg_ApplyConvolutionWindowAlongZ_kernel(float4 *smoothedImage,
                                                         int windowSize)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        int3 imageSize = c_ImageDim;

        short z=(int)(tid/((imageSize.x)*(imageSize.y)));

        int radius = (windowSize-1)/2;
        int index = tid - imageSize.x*imageSize.y*radius;
        z -= radius;

        float4 finalValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Kahan summation used here
        float3 c=make_float3(0.f,0.f,0.f), Y, t;
        float windowValue;
        for(int i=0; i<windowSize; i++){
            if(-1<z && z<imageSize.z){
                float4 gradientValue = tex1Dfetch(gradientImageTexture,index);
                windowValue = tex1Dfetch(convolutionKernelTexture,i);

                Y.x = gradientValue.x * windowValue - c.x;
                Y.y = gradientValue.y * windowValue - c.y;
                Y.z = gradientValue.z * windowValue - c.z;
                t.x = finalValue.x + Y.x;
                t.y = finalValue.y + Y.y;
                t.z = finalValue.z + Y.z;
                c.x = (t.x - finalValue.x) - Y.x;
                c.y = (t.y - finalValue.y) - Y.y;
                c.z = (t.z - finalValue.z) - Y.z;
                finalValue = make_float4(t.x, t.y, t.z, 0.f);
            }
            index += imageSize.x*imageSize.y;
            z++;
        }
        smoothedImage[tid] = finalValue;
    }
    return;
}
/* *************************************************************** */
__global__ void reg_multiplyValue_kernel_float(float *array_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        array_d[tid] *= c_Weight;
    }
}
/* *************************************************************** */
__global__ void reg_multiplyValue_kernel_float4(float4 *array_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        float4 temp = array_d[tid];
        array_d[tid] = make_float4(temp.x*c_Weight,temp.y*c_Weight,temp.z*c_Weight,temp.w*c_Weight);
    }
}
/* *************************************************************** */
__global__ void reg_addValue_kernel_float(float *array_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        array_d[tid] += c_Weight;
    }
}
/* *************************************************************** */
__global__ void reg_addValue_kernel_float4(float4 *array_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        float4 temp = array_d[tid];
        array_d[tid] = make_float4(temp.x+c_Weight,temp.y+c_Weight,temp.z+c_Weight,temp.w+c_Weight);
    }
}
/* *************************************************************** */
__global__ void reg_multiplyArrays_kernel_float(float *array1_d, float *array2_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        array1_d[tid] *= array2_d[tid];
    }
}
/* *************************************************************** */
__global__ void reg_multiplyArrays_kernel_float4(float4 *array1_d, float4 *array2_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        float4 a = array1_d[tid];
        float4 b = array1_d[tid];
        array1_d[tid] = make_float4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);
    }
}
/* *************************************************************** */
__global__ void reg_addArrays_kernel_float(float *array1_d, float *array2_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        array1_d[tid] += array2_d[tid];
    }
}
/* *************************************************************** */
__global__ void reg_addArrays_kernel_float4(float4 *array1_d, float4 *array2_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber){
        float4 a = array1_d[tid];
        float4 b = array1_d[tid];
        array1_d[tid] = make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
    }
}
/* *************************************************************** */
__global__ void reg_fillMaskArray_kernel(int *array1_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid < c_VoxelNumber)
        array1_d[tid] = tid;
}
/* *************************************************************** */

#endif

