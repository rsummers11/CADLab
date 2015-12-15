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

texture<float, 3, cudaReadModeElementType> floatingTexture;
texture<float4, 1, cudaReadModeElementType> floatingMatrixTexture;
texture<float4, 1, cudaReadModeElementType> deformationFieldTexture;
texture<int, 1, cudaReadModeElementType> maskTexture;
/* *************************************************************** */
__device__ __constant__ int3 c_FloatingDim;
__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ float c_PaddingValue;
__device__ __constant__ int c_ActiveVoxelNumber;
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_resampleImage2D_kernel(float *resultArray)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        //Get the real world deformation in the floating space
        const int tid2 = tex1Dfetch(maskTexture,tid);
        float4 realdeformation = tex1Dfetch(deformationFieldTexture,tid);

        //Get the voxel-based deformation in the floating space
        float2 voxeldeformation;
        float4 matrix = tex1Dfetch(floatingMatrixTexture,0);
        voxeldeformation.x =
                matrix.x*realdeformation.x +
                matrix.y*realdeformation.y +
                matrix.w;
        matrix = tex1Dfetch(floatingMatrixTexture,1);
        voxeldeformation.y =
                matrix.x*realdeformation.x +
                matrix.y*realdeformation.y +
                matrix.w;

        int3 floatingImageSize = c_FloatingDim;
        if( voxeldeformation.x>=0.0f && voxeldeformation.x<=floatingImageSize.x-1 &&
            voxeldeformation.y>=0.0f && voxeldeformation.y<=floatingImageSize.y-1 ){
            resultArray[tid2]=tex3D(floatingTexture, voxeldeformation.x+0.5f, voxeldeformation.y+0.5f, 0.5f);
        }
        else resultArray[tid2]=c_PaddingValue;
    }
}
/* *************************************************************** */
__global__ void reg_resampleImage3D_kernel(float *resultArray)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){
        const int tid2 = tex1Dfetch(maskTexture,tid);

        //Get the real world deformation in the floating space
        float4 realdeformation = tex1Dfetch(deformationFieldTexture,tid);

        //Get the voxel-based deformation in the floating space
        float3 voxeldeformation;
        float4 matrix = tex1Dfetch(floatingMatrixTexture,0);
        voxeldeformation.x =	matrix.x*realdeformation.x + matrix.y*realdeformation.y  +
                                matrix.z*realdeformation.z  +  matrix.w;
        matrix = tex1Dfetch(floatingMatrixTexture,1);
        voxeldeformation.y =	matrix.x*realdeformation.x + matrix.y*realdeformation.y  +
                                matrix.z*realdeformation.z  +  matrix.w;
        matrix = tex1Dfetch(floatingMatrixTexture,2);
        voxeldeformation.z =	matrix.x*realdeformation.x + matrix.y*realdeformation.y  +
                                matrix.z*realdeformation.z  +  matrix.w;

        int3 floatingImageSize = c_FloatingDim;
        if( voxeldeformation.x>=0.0f && voxeldeformation.x<=floatingImageSize.x-1 &&
            voxeldeformation.y>=0.0f && voxeldeformation.y<=floatingImageSize.y-1 &&
            voxeldeformation.z>=0.0f && voxeldeformation.z<=floatingImageSize.z-1 ){
            resultArray[tid2]=tex3D(floatingTexture, voxeldeformation.x+0.5f, voxeldeformation.y+0.5f, voxeldeformation.z+0.5f);
        }
        else resultArray[tid2]=c_PaddingValue;
    }
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_getImageGradient2D_kernel(float4 *gradientArray)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        //Get the real world deformation in the floating space
        float4 realdeformation = tex1Dfetch(deformationFieldTexture,tid);

        //Get the voxel-based deformation in the floating space
        float3 voxeldeformation;
        float4 matrix = tex1Dfetch(floatingMatrixTexture,0);
        voxeldeformation.x =
                matrix.x*realdeformation.x +
                matrix.y*realdeformation.y  +
                matrix.w;
        matrix = tex1Dfetch(floatingMatrixTexture,1);
        voxeldeformation.y =
                matrix.x*realdeformation.x +
                matrix.y*realdeformation.y  +
                matrix.w;

        int2 voxel;
        voxel.x = (int)(voxeldeformation.x);
        voxel.y = (int)(voxeldeformation.y);

        float xBasis[2];
        float relative = fabsf(voxeldeformation.x - (float)voxel.x);
        xBasis[0]=1.0f-relative;
        xBasis[1]=relative;
        float yBasis[2];
        relative = fabsf(voxeldeformation.y - (float)voxel.y);
        yBasis[0]=1.0f-relative;
        yBasis[1]=relative;
        float deriv[2];
        deriv[0]=-1.0f;
        deriv[1]=1.0f;

        float4 gradientValue=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float2 relativedeformation;
        for(short b=0; b<2; b++){
            float2 tempValueX=make_float2(0.0f, 0.0f);
            relativedeformation.y=((float)voxel.y+(float)b+0.5f)/(float)c_FloatingDim.y;
            for(short a=0; a<2; a++){
                relativedeformation.x=((float)voxel.x+(float)a+0.5f)/(float)c_FloatingDim.x;
                float intensity=c_PaddingValue;

                if(0.f<=relativedeformation.x && relativedeformation.x<=1.f &&
                   0.f<=relativedeformation.y && relativedeformation.y<=1.f)
                    intensity=tex3D(floatingTexture,
                                    relativedeformation.x,
                                    relativedeformation.y,
                                    0.5f);

                tempValueX.x +=  intensity * deriv[a];
                tempValueX.y +=  intensity * xBasis[a];
            }
            gradientValue.x += tempValueX.x * yBasis[b];
            gradientValue.y += tempValueX.y * deriv[b];
        }
        gradientArray[tid]=gradientValue;
    }
}
/* *************************************************************** */
__global__ void reg_getImageGradient3D_kernel(float4 *gradientArray)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        //Get the real world deformation in the floating space
        float4 realdeformation = tex1Dfetch(deformationFieldTexture,tid);

        //Get the voxel-based deformation in the floating space
        float3 voxeldeformation;
        float4 matrix = tex1Dfetch(floatingMatrixTexture,0);
        voxeldeformation.x =	matrix.x*realdeformation.x + matrix.y*realdeformation.y  +
                                matrix.z*realdeformation.z  +  matrix.w;
        matrix = tex1Dfetch(floatingMatrixTexture,1);
        voxeldeformation.y =	matrix.x*realdeformation.x + matrix.y*realdeformation.y  +
                                matrix.z*realdeformation.z  +  matrix.w;
        matrix = tex1Dfetch(floatingMatrixTexture,2);
        voxeldeformation.z =	matrix.x*realdeformation.x + matrix.y*realdeformation.y  +
                                matrix.z*realdeformation.z  +  matrix.w;

        int3 voxel;
        voxel.x = (int)(voxeldeformation.x);
        voxel.y = (int)(voxeldeformation.y);
        voxel.z = (int)(voxeldeformation.z);

        float xBasis[2];
        float relative = fabsf(voxeldeformation.x - (float)voxel.x);
        xBasis[0]=1.0f-relative;
        xBasis[1]=relative;
        float yBasis[2];
        relative = fabsf(voxeldeformation.y - (float)voxel.y);
        yBasis[0]=1.0f-relative;
        yBasis[1]=relative;
        float zBasis[2];
        relative = fabsf(voxeldeformation.z - (float)voxel.z);
        zBasis[0]=1.0f-relative;
        zBasis[1]=relative;
        float deriv[2];
        deriv[0]=-1.0f;
        deriv[1]=1.0f;

        float4 gradientValue=make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 relativedeformation;
        for(short c=0; c<2; c++){
            relativedeformation.z=((float)voxel.z+(float)c+0.5f)/(float)c_FloatingDim.z;
            float3 tempValueY=make_float3(0.0f, 0.0f, 0.0f);
            for(short b=0; b<2; b++){
                float2 tempValueX=make_float2(0.0f, 0.0f);
                relativedeformation.y=((float)voxel.y+(float)b+0.5f)/(float)c_FloatingDim.y;
                for(short a=0; a<2; a++){
                    relativedeformation.x=((float)voxel.x+(float)a+0.5f)/(float)c_FloatingDim.x;
                    float intensity=c_PaddingValue;

                    if(0.f<=relativedeformation.x && relativedeformation.x<=1.f &&
                       0.f<=relativedeformation.y && relativedeformation.y<=1.f &&
                       0.f<=relativedeformation.z && relativedeformation.z<=1.f)
                        intensity=tex3D(floatingTexture,
                                        relativedeformation.x,
                                        relativedeformation.y,
                                        relativedeformation.z);

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
}
/* *************************************************************** */
/* *************************************************************** */
#endif
