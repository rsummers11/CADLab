/*
 *  _reg_resampling_kernels.cu
 *  
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */



#ifndef _REG_RESAMPLING_KERNELS_CU
#define _REG_RESAMPLING_KERNELS_CU

#include "stdio.h"

texture<float, 3, cudaReadModeElementType> sourceTexture;
texture<float4, 1, cudaReadModeElementType> sourceMatrixTexture;
texture<float4, 1, cudaReadModeElementType> positionFieldTexture;
texture<int, 1, cudaReadModeElementType> maskTexture;
/* *************************************************************** */
__device__ __constant__ int3 c_SourceDim;
__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ float c_PaddingValue;
__device__ __constant__ int c_ActiveVoxelNumber;
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_resampleSourceImage_kernel(float *resultArray)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        //Get the real world position in the source space
        float4 realPosition = tex1Dfetch(positionFieldTexture,tid);

        //Get the voxel-based position in the source space
        float3 voxelPosition;
        float4 matrix = tex1Dfetch(sourceMatrixTexture,0);
        voxelPosition.x =	matrix.x*realPosition.x + matrix.y*realPosition.y  +
                                matrix.z*realPosition.z  +  matrix.w;
        matrix = tex1Dfetch(sourceMatrixTexture,1);
        voxelPosition.y =	matrix.x*realPosition.x + matrix.y*realPosition.y  +
                                matrix.z*realPosition.z  +  matrix.w;
        matrix = tex1Dfetch(sourceMatrixTexture,2);
        voxelPosition.z =	matrix.x*realPosition.x + matrix.y*realPosition.y  +
                                matrix.z*realPosition.z  +  matrix.w;

        int3 sourceImageSize = c_SourceDim;
        float3 relativePosition;
        relativePosition.x=(voxelPosition.x+0.5f)/(float)sourceImageSize.x;
        relativePosition.y=(voxelPosition.y+0.5f)/(float)sourceImageSize.y;
        relativePosition.z=(voxelPosition.z+0.5f)/(float)sourceImageSize.z;
        if( relativePosition.x>=0.0f && relativePosition.x<=1.0f &&
            relativePosition.y>=0.0f && relativePosition.y<=1.0f &&
            relativePosition.z>=0.0f && relativePosition.z<=1.0f ){
            resultArray[tex1Dfetch(maskTexture,tid)]=tex3D(sourceTexture, relativePosition.x, relativePosition.y, relativePosition.z);
        }
        else resultArray[tex1Dfetch(maskTexture,tid)]=c_PaddingValue;
    }
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_getSourceImageGradient_kernel(float4 *gradientArray)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        //Get the real world position in the source space
        float4 realPosition = tex1Dfetch(positionFieldTexture,tid);
	
        //Get the voxel-based position in the source space
        float3 voxelPosition;
        float4 matrix = tex1Dfetch(sourceMatrixTexture,0);
        voxelPosition.x =	matrix.x*realPosition.x + matrix.y*realPosition.y  +
                                matrix.z*realPosition.z  +  matrix.w;
        matrix = tex1Dfetch(sourceMatrixTexture,1);
        voxelPosition.y =	matrix.x*realPosition.x + matrix.y*realPosition.y  +
                                matrix.z*realPosition.z  +  matrix.w;
        matrix = tex1Dfetch(sourceMatrixTexture,2);
        voxelPosition.z =	matrix.x*realPosition.x + matrix.y*realPosition.y  +
                                matrix.z*realPosition.z  +  matrix.w;

        int3 sourceImageSize = c_SourceDim;
        if(	0.0f<=voxelPosition.x && voxelPosition.x<=float(sourceImageSize.x-1) &&
                0.0f<=voxelPosition.y && voxelPosition.y<=float(sourceImageSize.y-1) &&
                0.0f<=voxelPosition.z && voxelPosition.z<=float(sourceImageSize.z-1)){

            int3 voxel;
            voxel.x = (int)(voxelPosition.x);
            voxel.y = (int)(voxelPosition.y);
            voxel.z = (int)(voxelPosition.z);

            float xBasis[2];
            float relative = fabsf(voxelPosition.x - (float)voxel.x);
            xBasis[0]=1.0f-relative;
            xBasis[1]=relative;
            float yBasis[2];
            relative = fabsf(voxelPosition.y - (float)voxel.y);
            yBasis[0]=1.0f-relative;
            yBasis[1]=relative;
            float zBasis[2];
            relative = fabsf(voxelPosition.z - (float)voxel.z);
            zBasis[0]=1.0f-relative;
            zBasis[1]=relative;
            float deriv[2];
            deriv[0]=-1.0f;
            deriv[1]=1.0f;

            float4 gradientValue=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 relativePosition;
            for(short c=0; c<2; c++){
                relativePosition.z=((float)voxel.z+(float)c+0.5f)/(float)c_SourceDim.z;
                float3 tempValueY=make_float3(0.0f, 0.0f, 0.0f);
                for(short b=0; b<2; b++){
                    float2 tempValueX=make_float2(0.0f, 0.0f);
                    relativePosition.y=((float)voxel.y+(float)b+0.5f)/(float)c_SourceDim.y;
                    for(short a=0; a<2; a++){
                        relativePosition.x=((float)voxel.x+(float)a+0.5f)/(float)c_SourceDim.x;
                        float intensity=tex3D(sourceTexture, relativePosition.x, relativePosition.y, relativePosition.z);

                        tempValueX.x +=  intensity * deriv[a];
                        tempValueX.y +=  intensity * xBasis[a];
                    }
                    tempValueY.x += tempValueX.x * yBasis[b];
                    tempValueY.y += tempValueX.y * deriv[b];
                    tempValueY.z += tempValueX.y * yBasis[b];
                }
                gradientValue.x += tempValueY.x * zBasis[c];
                gradientValue.y += tempValueY.y * zBasis[c];
                gradientValue.z += tempValueY.z * deriv[c];
            }
            gradientArray[tid]=gradientValue;
        }
        else gradientArray[tid]=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}
/* *************************************************************** */
/* *************************************************************** */
#endif
