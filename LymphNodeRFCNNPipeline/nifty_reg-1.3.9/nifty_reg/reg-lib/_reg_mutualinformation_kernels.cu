/*
 *  _reg_mutualinformation_kernels.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_kernels_CU
#define _REG_MUTUALINFORMATION_kernels_CU

#include <stdio.h>

#define COEFF_L 0.16666666f
#define COEFF_C 0.66666666f
#define COEFF_B 0.83333333f

__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ int3 c_ImageSize;

// Bins: Need 4 values for max 4 channels.
__device__ __constant__ int c_firstTargetBin;
__device__ __constant__ int c_secondTargetBin;
__device__ __constant__ int c_firstResultBin;
__device__ __constant__ int c_secondResultBin;

__device__ __constant__ float4 c_Entropies;
__device__ __constant__ float c_NMI;
__device__ __constant__ int c_ActiveVoxelNumber;

texture<float, 3, cudaReadModeElementType> firstTargetImageTexture;
texture<float, 1, cudaReadModeElementType> firstResultImageTexture;
texture<float4, 1, cudaReadModeElementType> firstResultImageGradientTexture;
texture<float, 1, cudaReadModeElementType> histogramTexture;
texture<float4, 1, cudaReadModeElementType> gradientImageTexture;
texture<int, 1, cudaReadModeElementType> maskTexture;

/// Added for the multichannel stuff. We currently only support 2 target and 2 source channels.
/// So we need another texture for the second target and source channel respectively.
texture<float, 3, cudaReadModeElementType> secondTargetImageTexture;
texture<float, 1, cudaReadModeElementType> secondResultImageTexture;
texture<float4, 1, cudaReadModeElementType> secondResultImageGradientTexture;


__device__ float GetBasisSplineValue(float x)
{
    x=fabsf(x);
    float value=0.0f;
    if(x<2.0f)
        if(x<1.0f)
            value = 2.0f/3.0f + (0.5f*x-1.0f)*x*x;
        else{
            x-=2.0f;
            value = -x*x*x/6.0f;
    }
    return value;
}
__device__ float GetBasisSplineDerivativeValue(float ori)
{
    float x=fabsf(ori);
    float value=0.0f;
    if(x<2.0f)
        if(x<1.0f)
            value = (1.5f*x-2.0f)*ori;
        else{
            x-=2.0f;
            value = -0.5f * x * x;
            if(ori<0.0f)value =-value;
    }
    return value;
}

__global__ void reg_getVoxelBasedNMIGradientUsingPW_kernel(float4 *voxelNMIGradientArray_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        const int targetIndex = tex1Dfetch(maskTexture,tid);
        int tempIndex=targetIndex;
        const int z = tempIndex/(c_ImageSize.x*c_ImageSize.y);
        tempIndex  -= z*c_ImageSize.x*c_ImageSize.y;
        const int y = tempIndex/c_ImageSize.x;
        const int x = tempIndex - y*c_ImageSize.x;

        float targetImageValue = tex3D(firstTargetImageTexture,
                                       ((float)x+0.5f)/(float)c_ImageSize.x,
                                       ((float)y+0.5f)/(float)c_ImageSize.y,
                                       ((float)z+0.5f)/(float)c_ImageSize.z);
        float resultImageValue = tex1Dfetch(firstResultImageTexture,targetIndex);
        float4 resultImageGradient = tex1Dfetch(firstResultImageGradientTexture,tid);

        float4 gradValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // No computation is performed if any of the point is part of the background
        // The two is added because the image is resample between 2 and bin +2
        // if 64 bins are used the histogram will have 68 bins et the image will be between 2 and 65
        if( targetImageValue>0.0f &&
            resultImageValue>0.0f &&
            targetImageValue<c_firstTargetBin &&
            resultImageValue<c_firstResultBin &&
            targetImageValue==targetImageValue &&
            resultImageValue==resultImageValue){

            targetImageValue = floor(targetImageValue);
            resultImageValue = floor(resultImageValue);

            float3 resDeriv = make_float3(
                resultImageGradient.x,
                resultImageGradient.y,
                resultImageGradient.z);

            if( resultImageGradient.x==resultImageGradient.x &&
                resultImageGradient.y==resultImageGradient.y &&
                resultImageGradient.z==resultImageGradient.z){

                float jointEntropyDerivative_X = 0.0f;
                float movingEntropyDerivative_X = 0.0f;
                float fixedEntropyDerivative_X = 0.0f;

                float jointEntropyDerivative_Y = 0.0f;
                float movingEntropyDerivative_Y = 0.0f;
                float fixedEntropyDerivative_Y = 0.0f;

                float jointEntropyDerivative_Z = 0.0f;
                float movingEntropyDerivative_Z = 0.0f;
                float fixedEntropyDerivative_Z = 0.0f;

                for(int t=(int)(targetImageValue-1.0f); t<(int)(targetImageValue+2.0f); t++){
                    if(-1<t && t<c_firstTargetBin){
                        for(int r=(int)(resultImageValue-1.0f); r<(int)(resultImageValue+2.0f); r++){
                            if(-1<r && r<c_firstResultBin){
                                float commonValue = GetBasisSplineValue((float)t-targetImageValue) *
                                    GetBasisSplineDerivativeValue((float)r-resultImageValue);

                                float jointLog = tex1Dfetch(histogramTexture, r*c_firstResultBin+t);
                                float targetLog = tex1Dfetch(histogramTexture, c_firstTargetBin*c_firstResultBin+t);
                                float resultLog = tex1Dfetch(histogramTexture, c_firstTargetBin*c_firstResultBin+c_firstTargetBin+r);

                                float temp = commonValue * resDeriv.x;
                                jointEntropyDerivative_X -= temp * jointLog;
                                fixedEntropyDerivative_X -= temp * targetLog;
                                movingEntropyDerivative_X -= temp * resultLog;

                                temp = commonValue * resDeriv.y;
                                jointEntropyDerivative_Y -= temp * jointLog;
                                fixedEntropyDerivative_Y -= temp * targetLog;
                                movingEntropyDerivative_Y -= temp * resultLog;

                                temp = commonValue * resDeriv.z;
                                jointEntropyDerivative_Z -= temp * jointLog;
                                fixedEntropyDerivative_Z -= temp * targetLog;
                                movingEntropyDerivative_Z -= temp * resultLog;
                            } // O<t<bin
                        } // t
                    } // 0<r<bin
                } // r

                float NMI= c_NMI;
                float temp = c_Entropies.z;
                // (Marc) I removed the normalisation by the voxel number as each gradient has to be normalised in the same way
                gradValue.x = (fixedEntropyDerivative_X + movingEntropyDerivative_X - NMI * jointEntropyDerivative_X) / temp;
                gradValue.y = (fixedEntropyDerivative_Y + movingEntropyDerivative_Y - NMI * jointEntropyDerivative_Y) / temp;
                gradValue.z = (fixedEntropyDerivative_Z + movingEntropyDerivative_Z - NMI * jointEntropyDerivative_Z) / temp;

            }
        }
        voxelNMIGradientArray_d[targetIndex]=gradValue;

    }
    return;
}

// Multichannel NMI gradient. Hardcoded for 2x2 NMI channels.
__global__ void reg_getVoxelBasedNMIGradientUsingPW2x2_kernel(float4 *voxelNMIGradientArray_d)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){
        const int targetIndex = tex1Dfetch(maskTexture,tid);
        int tempIndex=targetIndex;
        const int z = tempIndex/(c_ImageSize.x*c_ImageSize.y);
        tempIndex  -= z*c_ImageSize.x*c_ImageSize.y;
        const int y = tempIndex/c_ImageSize.x;
        const int x = tempIndex - y*c_ImageSize.x;

        float4 voxelValues = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        voxelValues.x = tex3D(firstTargetImageTexture,
                              ((float)x+0.5f)/(float)c_ImageSize.x,
                              ((float)y+0.5f)/(float)c_ImageSize.y,
                              ((float)z+0.5f)/(float)c_ImageSize.z);
        voxelValues.x = tex3D(secondTargetImageTexture,
                              ((float)x+0.5f)/(float)c_ImageSize.x,
                              ((float)y+0.5f)/(float)c_ImageSize.y,
                              ((float)z+0.5f)/(float)c_ImageSize.z);
        voxelValues.z = tex1Dfetch(firstResultImageTexture,targetIndex);
        voxelValues.w = tex1Dfetch(secondResultImageTexture,targetIndex);

        float4 firstResultImageGradient = tex1Dfetch(firstResultImageGradientTexture,tid);
        float4 secondResultImageGradient = tex1Dfetch(secondResultImageGradientTexture,tid);
        float4 gradValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Could remove some tests (which are not really needed) to reduce register
        // count. They should be put in again at some point for completeness and generality.
        if (voxelValues.x == voxelValues.x &&
            voxelValues.y == voxelValues.y &&
            voxelValues.z == voxelValues.z &&
            voxelValues.w == voxelValues.w &&
            voxelValues.x >= 0.0f &&
            voxelValues.y >= 0.0f &&
            voxelValues.z >= 0.0f &&
            voxelValues.w >= 0.0f &&
            voxelValues.x < c_firstTargetBin &&
            voxelValues.y < c_secondTargetBin &&
            voxelValues.z < c_firstResultBin &&
            voxelValues.w < c_secondResultBin)
        {
            voxelValues.x = (float)((int)voxelValues.x);
            voxelValues.y = (float)((int)voxelValues.y);
            voxelValues.z = (float)((int)voxelValues.z);
            voxelValues.w = (float)((int)voxelValues.w);

            if( firstResultImageGradient.x==firstResultImageGradient.x &&
                firstResultImageGradient.y==firstResultImageGradient.y &&
                firstResultImageGradient.z==firstResultImageGradient.z &&
                secondResultImageGradient.x==secondResultImageGradient.x &&
                secondResultImageGradient.y==secondResultImageGradient.y &&
                secondResultImageGradient.z==secondResultImageGradient.z)
            {
                float jointEntropyDerivative_X = 0.0f;
                float movingEntropyDerivative_X = 0.0f;
                float fixedEntropyDerivative_X = 0.0f;

                float jointEntropyDerivative_Y = 0.0f;
                float movingEntropyDerivative_Y = 0.0f;
                float fixedEntropyDerivative_Y = 0.0f;

                float jointEntropyDerivative_Z = 0.0f;
                float movingEntropyDerivative_Z = 0.0f;
                float fixedEntropyDerivative_Z = 0.0f;

                float jointLog, targetLog, resultLog, temp;
                float4 relative_pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                float s_x, s_y, s_z, s_w;
                float common_target_value = 0.0f;
                int target_flat_index, result_flat_index, total_target_entries, num_probabilities;
                for (int i=-1; i<2; ++i) {
                    relative_pos.x = (int)(voxelValues.x+i);

                    if (-1<relative_pos.x && relative_pos.x<c_firstTargetBin) {
                        for (int j=-1; j<2; ++j) {
                            relative_pos.y = (int)(voxelValues.y+j);

                            if (-1<relative_pos.y && relative_pos.y<c_secondTargetBin) {
                                s_x = GetBasisSplineValue(relative_pos.x-voxelValues.x);
                                s_y = GetBasisSplineValue(relative_pos.y-voxelValues.y);
                                common_target_value =  s_x * s_y;

                                for (int k=-1; k<2; ++k) {
                                    relative_pos.z = (int)(voxelValues.z+k);
                                    if (-1<relative_pos.z && relative_pos.z<c_firstResultBin) {
                                        s_x = GetBasisSplineDerivativeValue(relative_pos.z-voxelValues.z);
                                        s_w = GetBasisSplineValue(relative_pos.z-voxelValues.z);
                                        for (int l=-1; l<2; ++l) {
                                            relative_pos.w = (int)(voxelValues.w+l);
                                            if (-1<relative_pos.w && relative_pos.w<c_secondResultBin) {
                                                target_flat_index = relative_pos.x + relative_pos.y * c_firstTargetBin;
                                                result_flat_index = relative_pos.z + relative_pos.w * c_firstResultBin;
                                                total_target_entries = c_firstTargetBin * c_secondTargetBin;
                                                num_probabilities = total_target_entries * c_firstResultBin * c_secondResultBin;

                                                jointLog = tex1Dfetch(histogramTexture, target_flat_index + (result_flat_index * total_target_entries));
                                                targetLog = tex1Dfetch(histogramTexture, num_probabilities + target_flat_index);
                                                resultLog = tex1Dfetch(histogramTexture, num_probabilities + total_target_entries + result_flat_index);

                                                // Contribution from floating images. These arithmetic operations use
                                                // a lot of registers. Need to look into whether this can be reduced somehow.
                                                s_y = GetBasisSplineValue(relative_pos.w-voxelValues.w);
                                                s_z = GetBasisSplineDerivativeValue(relative_pos.w-voxelValues.w);
                                                temp = (s_x * firstResultImageGradient.x * s_y) +
                                                       (s_z * secondResultImageGradient.x * s_w);
                                                temp *= common_target_value;

                                                jointEntropyDerivative_X -= temp * jointLog;
                                                fixedEntropyDerivative_X -= temp * targetLog;
                                                movingEntropyDerivative_X -= temp * resultLog;

                                                temp = (s_x * firstResultImageGradient.y * s_y) +
                                                       (s_z * secondResultImageGradient.y * s_w);
                                                temp *= common_target_value;
                                                jointEntropyDerivative_Y -= temp * jointLog;
                                                fixedEntropyDerivative_Y -= temp * targetLog;
                                                movingEntropyDerivative_Y -= temp * resultLog;

                                                temp = (s_x * firstResultImageGradient.z * s_y) +
                                                       (s_z * secondResultImageGradient.z * s_w);
                                                temp *= common_target_value;
                                                jointEntropyDerivative_Z -= temp * jointLog;
                                                fixedEntropyDerivative_Z -= temp * targetLog;
                                                movingEntropyDerivative_Z -= temp * resultLog;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                gradValue.x = (fixedEntropyDerivative_X + movingEntropyDerivative_X - c_NMI * jointEntropyDerivative_X) / c_Entropies.z;
                gradValue.y = (fixedEntropyDerivative_Y + movingEntropyDerivative_Y - c_NMI * jointEntropyDerivative_Y) / c_Entropies.z;
                gradValue.z = (fixedEntropyDerivative_Z + movingEntropyDerivative_Z - c_NMI * jointEntropyDerivative_Z) / c_Entropies.z;
            }
        }
        voxelNMIGradientArray_d[targetIndex]=gradValue;
    }
}

__global__ void reg_smoothJointHistogramX_kernel(float *tempHistogram)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_secondTargetBin*c_firstResultBin*c_secondResultBin){
        // The starting index is computed
        unsigned int startingPoint=tid*c_firstTargetBin;
        unsigned int finishPoint=startingPoint+c_firstTargetBin;

        // The first point is computed
        tempHistogram[startingPoint] = (tex1Dfetch(histogramTexture, startingPoint) * COEFF_C +
                                       tex1Dfetch(histogramTexture, startingPoint+1) * COEFF_L) / COEFF_B;
        // The middle points are computed
        for(unsigned int i=startingPoint+1; i<finishPoint-1; ++i){
            tempHistogram[i] = tex1Dfetch(histogramTexture, i-1) * COEFF_L +
                               tex1Dfetch(histogramTexture, i) * COEFF_C +
                               tex1Dfetch(histogramTexture, i+1) * COEFF_L;
        }
        // The last point is computed
        tempHistogram[finishPoint-1] = (tex1Dfetch(histogramTexture, finishPoint-2) * COEFF_L +
                                       tex1Dfetch(histogramTexture, finishPoint-1) * COEFF_C) / COEFF_B;
    }
    return;
}

__global__ void reg_smoothJointHistogramY_kernel(float *tempHistogram)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_firstTargetBin*c_firstResultBin*c_secondResultBin){
        // The starting index is computed
        unsigned int startingPoint=tid + c_firstTargetBin*(c_secondTargetBin-1)*(c_firstResultBin*(int)(tid/(c_firstTargetBin*c_firstResultBin)) +
                                   (int)(tid/c_firstTargetBin - c_firstResultBin * (int)(tid/(c_firstTargetBin*c_firstResultBin))));
        unsigned int increment = c_firstTargetBin;
        unsigned int finishPoint=startingPoint+increment*c_secondTargetBin;

        // The first point is computed
        tempHistogram[startingPoint] = (tex1Dfetch(histogramTexture, startingPoint) * COEFF_C +
                                       tex1Dfetch(histogramTexture, startingPoint+increment) * COEFF_L) / COEFF_B;
        // The middle points are computed
        for(unsigned int i=startingPoint+increment; i<finishPoint-increment; i+=increment){
            tempHistogram[i] = tex1Dfetch(histogramTexture, i-increment) * COEFF_L +
                               tex1Dfetch(histogramTexture, i) * COEFF_C +
                               tex1Dfetch(histogramTexture, i+increment) * COEFF_L;
        }
        // The last point is computed
        tempHistogram[finishPoint-increment] = (tex1Dfetch(histogramTexture, finishPoint-2*increment) * COEFF_L +
                                       tex1Dfetch(histogramTexture, finishPoint-increment) * COEFF_C) / COEFF_B;
    }
    return;
}

__global__ void reg_smoothJointHistogramZ_kernel(float *tempHistogram)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_firstTargetBin*c_secondTargetBin*c_secondResultBin){
        // The starting index is computed
        unsigned int startingPoint=tid+c_firstTargetBin*c_secondTargetBin*(c_firstResultBin-1)*(int)(tid/(c_firstTargetBin*c_secondTargetBin));
        unsigned int increment = c_firstTargetBin*c_secondTargetBin;
        unsigned int finishPoint=startingPoint+increment*c_firstResultBin;

        // The first point is computed
        tempHistogram[startingPoint] = (tex1Dfetch(histogramTexture, startingPoint) * COEFF_C +
                                       tex1Dfetch(histogramTexture, startingPoint+increment) * COEFF_L) / COEFF_B;
        // The middle points are computed
        for(unsigned int i=startingPoint+increment; i<finishPoint-increment; i+=increment){
            tempHistogram[i] = tex1Dfetch(histogramTexture, i-increment) * COEFF_L +
                               tex1Dfetch(histogramTexture, i) * COEFF_C +
                               tex1Dfetch(histogramTexture, i+increment) * COEFF_L;
        }
        // The last point is computed
        tempHistogram[finishPoint-increment] = (tex1Dfetch(histogramTexture, finishPoint-2*increment) * COEFF_L +
                                       tex1Dfetch(histogramTexture, finishPoint-increment) * COEFF_C) / COEFF_B;
    }
    return;
}

__global__ void reg_smoothJointHistogramW_kernel(float *tempHistogram)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_firstTargetBin*c_secondTargetBin*c_firstResultBin){
        // The starting index is computed
        unsigned int startingPoint=tid;
        unsigned int increment = c_firstTargetBin*c_secondTargetBin*c_firstResultBin;
        unsigned int finishPoint=increment*c_secondResultBin;

        // The first point is computed
        tempHistogram[startingPoint] = (tex1Dfetch(histogramTexture, startingPoint) * COEFF_C +
                                       tex1Dfetch(histogramTexture, startingPoint+increment) * COEFF_L) / COEFF_B;
        // The middle points are computed
        for(unsigned int i=startingPoint+increment; i<finishPoint-increment; i+=increment){
            tempHistogram[i] = tex1Dfetch(histogramTexture, i-increment) * COEFF_L +
                               tex1Dfetch(histogramTexture, i) * COEFF_C +
                               tex1Dfetch(histogramTexture, i+increment) * COEFF_L;
        }
        // The last point is computed
        tempHistogram[finishPoint-increment] = (tex1Dfetch(histogramTexture, finishPoint-2*increment) * COEFF_L +
                                       tex1Dfetch(histogramTexture, finishPoint-increment) * COEFF_C) / COEFF_B;
    }
    return;
}

/// Kernels for marginalisation along the different axes
__global__ void reg_marginaliseTargetX_kernel(float *babyHisto)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_secondTargetBin*c_firstResultBin*c_secondResultBin){
        unsigned int startingPoint=tid*c_firstTargetBin;
        unsigned int finishPoint=startingPoint+c_firstTargetBin;

        float sum=tex1Dfetch(histogramTexture, startingPoint);
        float c=0.f,Y,t;
        for(unsigned int i=startingPoint+1; i<finishPoint; ++i){
            Y = tex1Dfetch(histogramTexture, i) - c;
            t = sum + Y;
            c = (t-sum)-Y;
            sum=t;
        }
        babyHisto[tid]=sum;
    }
}

__global__ void reg_marginaliseTargetXY_kernel(float *babyHisto)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_firstResultBin*c_secondResultBin){
        unsigned int startingPoint=tid*c_secondTargetBin;
        unsigned int finishPoint=startingPoint+c_secondTargetBin;

        float sum=tex1Dfetch(histogramTexture, startingPoint);        
        float c=0.f,Y,t;
        for(unsigned int i=startingPoint+1; i<finishPoint; ++i){            
            Y = tex1Dfetch(histogramTexture, i) - c;
            t = sum + Y;
            c = (t-sum)-Y;
            sum=t;
        }        
        babyHisto[tid]=sum;
    }
}

__global__ void reg_marginaliseResultX_kernel(float *babyHisto)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_firstTargetBin*c_secondTargetBin*c_firstResultBin){
        unsigned int startingPoint = tid;
        float sum=tex1Dfetch(histogramTexture, startingPoint);
        // increment by a the cube
        unsigned int increment = c_firstTargetBin*c_secondTargetBin*c_firstResultBin;
        float c=0.f,Y,t;

        for (unsigned int i = 1; i < c_secondResultBin; ++i)
        {
            Y = tex1Dfetch(histogramTexture, startingPoint + i *increment) - c;
            t = sum + Y;
            c = (t-sum)-Y;
            sum=t;
        }
        babyHisto[tid]=sum;
    }
}

__global__ void reg_marginaliseResultXY_kernel(float *babyHisto)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_firstTargetBin*c_secondTargetBin){
        unsigned int startingPoint=tid;
        float sum=tex1Dfetch(histogramTexture, startingPoint);
        // increment by the plane.
        unsigned int increment = c_firstTargetBin*c_secondTargetBin;
        float c=0.f,Y,t;
        for (unsigned int i = 1; i < c_firstResultBin; ++i)
        {
            Y = tex1Dfetch(histogramTexture, startingPoint + i *increment) - c;
            t = sum + Y;
            c = (t-sum)-Y;
            sum=t;
        }
        babyHisto[tid]=sum;
    }
}

#endif
