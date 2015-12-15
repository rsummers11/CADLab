/*
 * @file _reg_ssd_kernels.cu
 * @author Marc Modat
 * @date 14/11/2012
 *
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_SSD_KERNELS_CU
#define _REG_SSD_KERNELS_CU

#include "_reg_ssd_gpu.h"
#include "_reg_ssd_kernels.cu"
/* *************************************************************** */
texture<float, 3, cudaReadModeElementType> referenceTexture;
texture<float, 1, cudaReadModeElementType> warpedTexture;
texture<int, 1, cudaReadModeElementType> maskTexture;
texture<float4, 1, cudaReadModeElementType> spaGradientTexture;
/* *************************************************************** */
__device__ __constant__ int c_ActiveVoxelNumber;
__device__ __constant__ int3 c_ReferenceImageDim;
__device__ __constant__ float c_NormalisationNumber;
/* *************************************************************** */
__global__ void reg_getSquaredDifference3D_kernel(float *squaredDifference)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        int3 imageSize = c_ReferenceImageDim;
        unsigned int index=tex1Dfetch(maskTexture,tid);
        const int z = index/(imageSize.x*imageSize.y);
        const int tempIndex = index - z*imageSize.x*imageSize.y;
        const int y = tempIndex/imageSize.x;
        const int x = tempIndex - y*imageSize.x;

        float difference = tex3D(referenceTexture,
                                    ((float)x+0.5f)/(float)imageSize.x,
                                    ((float)y+0.5f)/(float)imageSize.y,
                                    ((float)z+0.5f)/(float)imageSize.z);
        difference -= tex1Dfetch(warpedTexture,index);
        if(difference==difference)
            squaredDifference[tid]= difference*difference;
        else squaredDifference[tid] = 0.f;
    }
}
/* *************************************************************** */
__global__ void reg_getSquaredDifference2D_kernel(float *squaredDifference)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        int3 imageSize = c_ReferenceImageDim;
        unsigned int index=tex1Dfetch(maskTexture,tid);
        const int y = index/imageSize.x;
        const int x = index - y*imageSize.x;

        float difference = tex3D(referenceTexture,
                                    ((float)x+0.5f)/(float)imageSize.x,
                                    ((float)y+0.5f)/(float)imageSize.y,
                                    0.5f);
        difference -= tex1Dfetch(warpedTexture,index);
        if(difference==difference)
            squaredDifference[tid]= difference*difference;
        else squaredDifference[tid] = 0.f;
    }
}
/* *************************************************************** */
__global__ void reg_getSSDGradient2D_kernel(float4 *ssdGradient)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        int3 imageSize = c_ReferenceImageDim;
        unsigned int index = tex1Dfetch(maskTexture,tid);
        const int y = index/imageSize.x;
        const int x = index - y*imageSize.x;

        float refValue = tex3D(referenceTexture,
                               ((float)x+0.5f)/(float)imageSize.x,
                               ((float)y+0.5f)/(float)imageSize.y,
                               0.5f);
        if(refValue != refValue)
            return;
        float warpValue = tex1Dfetch(warpedTexture,index);
        if(warpValue != warpValue)
            return;

        float4 spaGradientValue = tex1Dfetch(spaGradientTexture,tid);
        if(spaGradientValue.x != spaGradientValue.x ||
           spaGradientValue.y != spaGradientValue.y)
            return;

        float common = -2.f * (refValue - warpValue) /
                (c_NormalisationNumber * (float)c_ActiveVoxelNumber);

        ssdGradient[index] = make_float4(
                    common * spaGradientValue.x,
                    common * spaGradientValue.y,
                    0.f,
                    0.f
                    );
    }
}
/* *************************************************************** */
__global__ void reg_getSSDGradient3D_kernel(float4 *ssdGradient)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        int3 imageSize = c_ReferenceImageDim;
        unsigned int index = tex1Dfetch(maskTexture,tid);
        const int z = index/(imageSize.x*imageSize.y);
        const int tempIndex = index - z*imageSize.x*imageSize.y;
        const int y = tempIndex/imageSize.x;
        const int x = tempIndex - y*imageSize.x;

        float refValue = tex3D(referenceTexture,
                               ((float)x+0.5f)/(float)imageSize.x,
                               ((float)y+0.5f)/(float)imageSize.y,
                               ((float)z+0.5f)/(float)imageSize.z);
        if(refValue != refValue)
            return;

        float warpValue = tex1Dfetch(warpedTexture,index);
        if(warpValue != warpValue)
            return;

        float4 spaGradientValue = tex1Dfetch(spaGradientTexture,tid);
        if(spaGradientValue.x != spaGradientValue.x ||
           spaGradientValue.y != spaGradientValue.y ||
           spaGradientValue.z != spaGradientValue.z)
            return;

        float common = -2.f * (refValue - warpValue) /
                (c_NormalisationNumber * (float)c_ActiveVoxelNumber);

        ssdGradient[index] = make_float4(
                    common * spaGradientValue.x,
                    common * spaGradientValue.y,
                    common * spaGradientValue.z,
                    0.f
                    );
    }
}
/* *************************************************************** */
#endif

