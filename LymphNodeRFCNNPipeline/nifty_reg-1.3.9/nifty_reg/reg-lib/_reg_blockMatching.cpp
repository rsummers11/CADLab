/*
 *  _reg_blockMatching.cpp
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_blockMatching.h"
#include "_reg_globalTransformation.h"
#include <map>
#include <iostream>
#include <limits>



/* *************************************************************** */
/* *************************************************************** */
void copy_transformation_4x4(const mat44 & source, mat44 & dest)
{
    dest.m[0][0] = source.m[0][0];
    dest.m[0][1] = source.m[0][1];
    dest.m[0][2] = source.m[0][2];
    dest.m[0][3] = source.m[0][3];

    dest.m[1][0] = source.m[1][0];
    dest.m[1][1] = source.m[1][1];
    dest.m[1][2] = source.m[1][2];
    dest.m[1][3] = source.m[1][3];

    dest.m[2][0] = source.m[2][0];
    dest.m[2][1] = source.m[2][1];
    dest.m[2][2] = source.m[2][2];
    dest.m[2][3] = source.m[2][3];

    dest.m[3][0] = source.m[3][0];
    dest.m[3][1] = source.m[3][1];
    dest.m[3][2] = source.m[3][2];
    dest.m[3][3] = source.m[3][3];
}
/* *************************************************************** */
/* *************************************************************** */
// Helper function: Get the square of the Euclidean distance
double get_square_distance(float * first_point3D, float * second_point3D)
{
    return  sqrt((first_point3D[0]-second_point3D[0])*(first_point3D[0]-second_point3D[0]) +
                 (first_point3D[1]-second_point3D[1])*(first_point3D[1]-second_point3D[1]) +
                 (first_point3D[2]-second_point3D[2])*(first_point3D[2]-second_point3D[2]));
}
double get_square_distance2D(float * first_point2D, float * second_point2D)
{
    return  sqrt((first_point2D[0]-second_point2D[0])*(first_point2D[0]-second_point2D[0]) +
                 (first_point2D[1]-second_point2D[1])*(first_point2D[1]-second_point2D[1]));
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
        void _reg_set_active_blocks(nifti_image *targetImage, _reg_blockMatchingParam *params, int *mask, bool runningOnGPU)
{
    const int totalBlockNumber = params->blockNumber[0]*params->blockNumber[1]*params->blockNumber[2];
    float *varianceArray=(float *)malloc(totalBlockNumber*sizeof(float));
    int *indexArray=(int *)malloc(totalBlockNumber*sizeof(int));

    int *maskPtr=&mask[0];

    int unusableBlock=0;
    int index;

    DTYPE *targetValues = (DTYPE *)malloc(BLOCK_SIZE * sizeof(DTYPE));
    DTYPE *targetPtr = static_cast<DTYPE *>(targetImage->data);
    int blockIndex=0;

    if(targetImage->nz>1){
        // Version using 3D blocks
        for(int k=0; k<params->blockNumber[2]; k++){
            for(int j=0; j<params->blockNumber[1]; j++){
                for(int i=0; i<params->blockNumber[0]; i++){
                    for(unsigned int n=0; n<BLOCK_SIZE; n++)
                        targetValues[n]=(DTYPE)std::numeric_limits<float>::quiet_NaN();
                    float mean=0.0f;
                    float voxelNumber=0.0f;
                    int coord=0;
                    for(int z=k*BLOCK_WIDTH; z<(k+1)*BLOCK_WIDTH; z++){
                        if(z<targetImage->nz){
                            index =z*targetImage->nx*targetImage->ny;
                            DTYPE *targetPtrZ=&targetPtr[index];
                            int *maskPtrZ=&maskPtr[index];
                            for(int y=j*BLOCK_WIDTH; y<(j+1)*BLOCK_WIDTH; y++){
                                if(y<targetImage->ny){
                                    index = y*targetImage->nx+i*BLOCK_WIDTH;
                                    DTYPE *targetPtrXYZ=&targetPtrZ[index];
                                    int *maskPtrXYZ=&maskPtrZ[index];
                                    for(int x=i*BLOCK_WIDTH; x<(i+1)*BLOCK_WIDTH; x++){
                                        if(x<targetImage->nx){
                                            targetValues[coord] = *targetPtrXYZ;
                                            if(targetValues[coord]==targetValues[coord] && targetValues[coord]!=0. && *maskPtrXYZ>-1){
                                                mean += (float)targetValues[coord];
                                                voxelNumber++;
                                            }
                                        }
                                        targetPtrXYZ++;
                                        maskPtrXYZ++;
                                        coord++;
                                    }
                                }
                            }
                        }
                    }
                    if(voxelNumber>BLOCK_SIZE/2){
                        float variance=0.0f;
                        for(int i=0; i<BLOCK_SIZE; i++){
                            if(targetValues[i]==targetValues[i])
                                variance += (mean - (float)targetValues[i])
                                * (mean - (float)targetValues[i]);
                        }

                        variance /= voxelNumber;
                        varianceArray[blockIndex]=variance;
                    }
                    else{
                        varianceArray[blockIndex]=-1;
                        unusableBlock++;
                    }
                    indexArray[blockIndex]=blockIndex;
                    blockIndex++;
                }
            }
        }
    }
    else{
        // Version using 2D blocks
        for(int j=0; j<params->blockNumber[1]; j++){
            for(int i=0; i<params->blockNumber[0]; i++){

                for(unsigned int n=0; n<BLOCK_2D_SIZE; n++)
                    targetValues[n]=(DTYPE)std::numeric_limits<float>::quiet_NaN();
                float mean=0.0f;
                float voxelNumber=0.0f;
                int coord=0;

                for(int y=j*BLOCK_WIDTH; y<(j+1)*BLOCK_WIDTH; y++){
                    if(y<targetImage->ny){
                        index = y*targetImage->nx+i*BLOCK_WIDTH;
                        DTYPE *targetPtrXY=&targetPtr[index];
                        int *maskPtrXY=&maskPtr[index];
                        for(int x=i*BLOCK_WIDTH; x<(i+1)*BLOCK_WIDTH; x++){
                            if(x<targetImage->nx){
                                targetValues[coord] = *targetPtrXY;
                                if(targetValues[coord]==targetValues[coord] && targetValues[coord]!=0. && *maskPtrXY>-1){
                                    mean += (float)targetValues[coord];
                                    voxelNumber++;
                                }
                            }
                            targetPtrXY++;
                            maskPtrXY++;
                            coord++;
                        }
                    }
                }
                if(voxelNumber>BLOCK_2D_SIZE/2){
                    float variance=0.0f;
                    for(int i=0; i<BLOCK_2D_SIZE; i++){
                        if(targetValues[i]==targetValues[i])
                            variance += (mean - (float)targetValues[i])
                            * (mean - (float)targetValues[i]);
                    }

                    variance /= voxelNumber;
                    varianceArray[blockIndex]=variance;
                }
                else{
                    varianceArray[blockIndex]=-1;
                    unusableBlock++;
                }
                indexArray[blockIndex]=blockIndex;
                blockIndex++;
            }
        }
    }
    free(targetValues);

    params->activeBlockNumber=params->activeBlockNumber<(totalBlockNumber-unusableBlock)?params->activeBlockNumber:(totalBlockNumber-unusableBlock);

    reg_heapSort(varianceArray, indexArray, totalBlockNumber);

    memset(params->activeBlock, 0, totalBlockNumber * sizeof(int));
    int *indexArrayPtr = &indexArray[totalBlockNumber-1];
    int count = 0;
    for(int i=0; i<params->activeBlockNumber; i++){
        params->activeBlock[*indexArrayPtr--] = count++;
    }
    for (int i = params->activeBlockNumber; i < totalBlockNumber; ++i){
        params->activeBlock[*indexArrayPtr--] = -1;
    }

    count = 0;
    if (runningOnGPU) {
        for(int i = 0; i < totalBlockNumber; ++i){
            if(params->activeBlock[i] != -1){
                params->activeBlock[i] = -1;
                params->activeBlock[count] = i;
                ++count;
            }
        }
    }

    free(varianceArray);
    free(indexArray);
}
/* *************************************************************** */
void initialise_block_matching_method(  nifti_image * target,
                                        _reg_blockMatchingParam *params,
                                        int percentToKeep_block,
                                        int percentToKeep_opt,
                                        int *mask,
                                        bool runningOnGPU)
{
    params->blockNumber[0]=(int)ceil((float)target->nx / (float)BLOCK_WIDTH);
    params->blockNumber[1]=(int)ceil((float)target->ny / (float)BLOCK_WIDTH);
    if(target->nz>1)
        params->blockNumber[2]=(int)ceil((float)target->nz / (float)BLOCK_WIDTH);
    else params->blockNumber[2]=1;

    params->percent_to_keep=percentToKeep_opt;
    params->activeBlockNumber=params->blockNumber[0]*params->blockNumber[1]*params->blockNumber[2] * percentToKeep_block / 100;

    params->activeBlock = (int *)malloc(params->blockNumber[0]*params->blockNumber[1]*params->blockNumber[2] * sizeof(int));
    switch(target->datatype){
        case NIFTI_TYPE_FLOAT32:
            _reg_set_active_blocks<float>(target, params, mask, runningOnGPU);break;
        case NIFTI_TYPE_FLOAT64:
            _reg_set_active_blocks<double>(target, params, mask, runningOnGPU);break;
        default:
            fprintf(stderr,"[NiftyReg ERROR] initialise_block_matching_method\tThe target image data type is not supported\n");
            exit(1);
    }
    if(params->activeBlockNumber<2){
        fprintf(stderr,"[NiftyReg ERROR] There are no active blocks\n");
        fprintf(stderr,"[NiftyReg ERROR] ... Exit ...\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG]: There are %i active block(s) out of %i.\n", params->activeBlockNumber, params->blockNumber[0]*params->blockNumber[1]*params->blockNumber[2]);
#endif
    if(target->nz>1){
        params->targetPosition = (float *)malloc(params->activeBlockNumber*3*sizeof(float));
        params->resultPosition = (float *)malloc(params->activeBlockNumber*3*sizeof(float));
    }
    else{
        params->targetPosition = (float *)malloc(params->activeBlockNumber*2*sizeof(float));
        params->resultPosition = (float *)malloc(params->activeBlockNumber*2*sizeof(float));
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] block matching initialisation done.\n");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<typename PrecisionTYPE, typename TargetImageType, typename ResultImageType>
        void block_matching_method2D(nifti_image * target,
                                     nifti_image * result,
                                     _reg_blockMatchingParam *params,
                                     int *mask)
{
    TargetImageType *targetPtr=static_cast<TargetImageType *>(target->data);
    ResultImageType *resultPtr=static_cast<ResultImageType *>(result->data);

    TargetImageType *targetValues=(TargetImageType *)malloc(BLOCK_2D_SIZE*sizeof(TargetImageType));
    bool *targetOverlap=(bool *)malloc(BLOCK_2D_SIZE*sizeof(bool));
    ResultImageType *resultValues=(ResultImageType *)malloc(BLOCK_2D_SIZE*sizeof(ResultImageType));
    bool *resultOverlap=(bool *)malloc(BLOCK_2D_SIZE*sizeof(bool));

    mat44 *targetMatrix_xyz;
    if(target->sform_code >0)
        targetMatrix_xyz = &(target->sto_xyz);
    else targetMatrix_xyz = &(target->qto_xyz);

    int targetIndex_start_x;
    int targetIndex_start_y;
    int targetIndex_end_x;
    int targetIndex_end_y;
    int resultIndex_start_x;
    int resultIndex_start_y;
    int resultIndex_end_x;
    int resultIndex_end_y;

    unsigned int targetIndex;
    unsigned int resultIndex;

    unsigned int blockIndex=0;
    unsigned int activeBlockIndex=0;
    params->definedActiveBlock=0;
    int index;

    for(int j=0; j<params->blockNumber[1]; j++){
        targetIndex_start_y=j*BLOCK_WIDTH;
        targetIndex_end_y=targetIndex_start_y+BLOCK_WIDTH;

        for(int i=0; i<params->blockNumber[0]; i++){
            targetIndex_start_x=i*BLOCK_WIDTH;
            targetIndex_end_x=targetIndex_start_x+BLOCK_WIDTH;

            if(params->activeBlock[blockIndex] > -1){

                targetIndex=0;
                memset(targetOverlap, 0, BLOCK_2D_SIZE*sizeof(bool));

                for(int y=targetIndex_start_y; y<targetIndex_end_y; y++){
                    if(-1<y && y<target->ny){
                        index = y*target->nx+targetIndex_start_x;
                        TargetImageType *targetPtr_XY = &targetPtr[index];
                        int *maskPtr_XY=&mask[index];
                        for(int x=targetIndex_start_x; x<targetIndex_end_x; x++){
                            if(-1<x && x<target->nx){
                                TargetImageType value = *targetPtr_XY;
                                if(value==value && value!=0. && *maskPtr_XY>-1){
                                    targetValues[targetIndex]=value;
                                    targetOverlap[targetIndex]=1;
                                }
                            }
                            targetPtr_XY++;
                            maskPtr_XY++;
                            targetIndex++;
                        }
                    }
                    else targetIndex+=BLOCK_WIDTH;
                }
                PrecisionTYPE bestCC=0.0;
                float bestDisplacement[3] = {std::numeric_limits<float>::quiet_NaN(),
                                             0.f, 0.f};

                // iteration over the result blocks
                for(int m=-OVERLAP_SIZE; m<=OVERLAP_SIZE; m+=STEP_SIZE){
                    resultIndex_start_y=targetIndex_start_y+m;
                    resultIndex_end_y=resultIndex_start_y+BLOCK_WIDTH;
                    for(int l=-OVERLAP_SIZE; l<=OVERLAP_SIZE; l+=STEP_SIZE){
                        resultIndex_start_x=targetIndex_start_x+l;
                        resultIndex_end_x=resultIndex_start_x+BLOCK_WIDTH;

                        resultIndex=0;
                        memset(resultOverlap, 0, BLOCK_2D_SIZE*sizeof(bool));

                        for(int y=resultIndex_start_y; y<resultIndex_end_y; y++){
                            if(-1<y && y<result->ny){
                                index=y*result->nx+resultIndex_start_x;
                                ResultImageType *resultPtr_XY = &resultPtr[index];
                                int *maskPtr_XY=&mask[index];
                                for(int x=resultIndex_start_x; x<resultIndex_end_x; x++){
                                    if(-1<x && x<result->nx){
                                        ResultImageType value = *resultPtr_XY;
                                        if(value==value && value!=0. && *maskPtr_XY>-1){
                                            resultValues[resultIndex]=value;
                                            resultOverlap[resultIndex]=1;
                                        }
                                    }
                                    resultPtr_XY++;
                                    resultIndex++;
                                    maskPtr_XY++;
                                }
                            }
                            else resultIndex+=BLOCK_WIDTH;
                        }
                        PrecisionTYPE targetMean=0.0;
                        PrecisionTYPE resultMean=0.0;
                        PrecisionTYPE voxelNumber=0.0;
                        for(int a=0; a<BLOCK_2D_SIZE; a++){
                            if(targetOverlap[a] && resultOverlap[a]){
                                targetMean += (PrecisionTYPE)targetValues[a];
                                resultMean += (PrecisionTYPE)resultValues[a];
                                voxelNumber++;
                            }
                        }

                        if(voxelNumber>BLOCK_2D_SIZE/2){
                            targetMean /= voxelNumber;
                            resultMean /= voxelNumber;

                            PrecisionTYPE targetVar=0.0;
                            PrecisionTYPE resultVar=0.0;
                            PrecisionTYPE localCC=0.0;

                            for(int a=0; a<BLOCK_2D_SIZE; a++){
                                if(targetOverlap[a] && resultOverlap[a]){
                                    PrecisionTYPE targetTemp=(PrecisionTYPE)(targetValues[a]-targetMean);
                                    PrecisionTYPE resultTemp=(PrecisionTYPE)(resultValues[a]-resultMean);
                                    targetVar += (targetTemp)*(targetTemp);
                                    resultVar += (resultTemp)*(resultTemp);
                                    localCC += (targetTemp)*(resultTemp);
                                }
                            }

                            localCC = fabs(localCC/sqrt(targetVar*resultVar));

                            if(localCC>bestCC){
                                bestCC=localCC;
                                bestDisplacement[0] = (float)l;
                                bestDisplacement[1] = (float)m;
                            }
                        }
                    }
                }

                if(bestDisplacement[0]==bestDisplacement[0]){
                    float targetPosition_temp[3];
                    targetPosition_temp[0] = (float)(i*BLOCK_WIDTH);
                    targetPosition_temp[1] = (float)(j*BLOCK_WIDTH);
                    targetPosition_temp[2] = 0.0f;

                    bestDisplacement[0] += targetPosition_temp[0];
                    bestDisplacement[1] += targetPosition_temp[1];
                    bestDisplacement[2] = 0.0f;

                    float tempPosition[3];
                    apply_affine(targetMatrix_xyz, targetPosition_temp, tempPosition);
                    params->targetPosition[activeBlockIndex] = tempPosition[0];
                    params->targetPosition[activeBlockIndex+1] = tempPosition[1];
                    apply_affine(targetMatrix_xyz, bestDisplacement, tempPosition);
                    params->resultPosition[activeBlockIndex] = tempPosition[0];
                    params->resultPosition[activeBlockIndex+1] = tempPosition[1];
                    activeBlockIndex += 2;
                    params->definedActiveBlock++;
                }
            }
            blockIndex++;
        }
    }
    free(resultValues);
    free(targetValues);
    free(targetOverlap);
    free(resultOverlap);
}
/* *************************************************************** */
template<typename PrecisionTYPE, typename TargetImageType, typename ResultImageType>
        void block_matching_method3D(nifti_image * target,
                                     nifti_image * result,
                                     _reg_blockMatchingParam *params,
                                     int *mask)
{
    TargetImageType *targetPtr=static_cast<TargetImageType *>(target->data);
    ResultImageType *resultPtr=static_cast<ResultImageType *>(result->data);

    TargetImageType *targetValues=(TargetImageType *)malloc(BLOCK_SIZE*sizeof(TargetImageType));
    bool *targetOverlap=(bool *)malloc(BLOCK_SIZE*sizeof(bool));
    ResultImageType *resultValues=(ResultImageType *)malloc(BLOCK_SIZE*sizeof(ResultImageType));
    bool *resultOverlap=(bool *)malloc(BLOCK_SIZE*sizeof(bool));

    mat44 *targetMatrix_xyz;
    if(target->sform_code >0)
        targetMatrix_xyz = &(target->sto_xyz);
    else targetMatrix_xyz = &(target->qto_xyz);

    int targetIndex_start_x;
    int targetIndex_start_y;
    int targetIndex_start_z;
    int targetIndex_end_x;
    int targetIndex_end_y;
    int targetIndex_end_z;
    int resultIndex_start_x;
    int resultIndex_start_y;
    int resultIndex_start_z;
    int resultIndex_end_x;
    int resultIndex_end_y;
    int resultIndex_end_z;

    unsigned int targetIndex;
    unsigned int resultIndex;

    unsigned int blockIndex=0;
    unsigned int activeBlockIndex=0;
    int index;
    params->definedActiveBlock=0;

    for(int k=0; k<params->blockNumber[2]; k++){
        targetIndex_start_z=k*BLOCK_WIDTH;
        targetIndex_end_z=targetIndex_start_z+BLOCK_WIDTH;

        for(int j=0; j<params->blockNumber[1]; j++){
            targetIndex_start_y=j*BLOCK_WIDTH;
            targetIndex_end_y=targetIndex_start_y+BLOCK_WIDTH;

            for(int i=0; i<params->blockNumber[0]; i++){
                targetIndex_start_x=i*BLOCK_WIDTH;
                targetIndex_end_x=targetIndex_start_x+BLOCK_WIDTH;

                if(params->activeBlock[blockIndex] > -1){
                    targetIndex=0;
                    memset(targetOverlap, 0, BLOCK_SIZE*sizeof(bool));
                    for(int z=targetIndex_start_z; z<targetIndex_end_z; z++){
                        if(-1<z && z<target->nz){
                            index = z*target->nx*target->ny;
                            TargetImageType *targetPtr_Z = &targetPtr[index];
                            int *maskPtr_Z=&mask[index];
                            for(int y=targetIndex_start_y; y<targetIndex_end_y; y++){
                                if(-1<y && y<target->ny){
                                    index = y*target->nx+targetIndex_start_x;
                                    TargetImageType *targetPtr_XYZ = &targetPtr_Z[index];
                                    int *maskPtr_XYZ=&maskPtr_Z[index];
                                    for(int x=targetIndex_start_x; x<targetIndex_end_x; x++){
                                        if(-1<x && x<target->nx){
                                            TargetImageType value = *targetPtr_XYZ;
                                            if(value==value && *maskPtr_XYZ>-1){
                                                targetValues[targetIndex]=value;
                                                targetOverlap[targetIndex]=1;
                                            }
                                        }
                                        targetPtr_XYZ++;
                                        maskPtr_XYZ++;
                                        targetIndex++;
                                    }
                                }
                                else targetIndex+=BLOCK_WIDTH;
                            }
                        }
                        else targetIndex+=BLOCK_WIDTH*BLOCK_WIDTH;
                    }
                    PrecisionTYPE bestCC=0.0;
                    float bestDisplacement[3] = {std::numeric_limits<float>::quiet_NaN(),
                                                 0.f, 0.f};

                    // iteration over the result blocks
                    for(int n=-OVERLAP_SIZE; n<=OVERLAP_SIZE; n+=STEP_SIZE){
                        resultIndex_start_z=targetIndex_start_z+n;
                        resultIndex_end_z=resultIndex_start_z+BLOCK_WIDTH;
                        for(int m=-OVERLAP_SIZE; m<=OVERLAP_SIZE; m+=STEP_SIZE){
                            resultIndex_start_y=targetIndex_start_y+m;
                            resultIndex_end_y=resultIndex_start_y+BLOCK_WIDTH;
                            for(int l=-OVERLAP_SIZE; l<=OVERLAP_SIZE; l+=STEP_SIZE){
                                resultIndex_start_x=targetIndex_start_x+l;
                                resultIndex_end_x=resultIndex_start_x+BLOCK_WIDTH;

                                resultIndex=0;
                                memset(resultOverlap, 0, BLOCK_SIZE*sizeof(bool));

                                for(int z=resultIndex_start_z; z<resultIndex_end_z; z++){
                                    if(-1<z && z<result->nz){
                                        index = z*result->nx*result->ny;
                                        ResultImageType *resultPtr_Z = &resultPtr[index];
                                        int *maskPtr_Z = &mask[index];
                                        for(int y=resultIndex_start_y; y<resultIndex_end_y; y++){
                                            if(-1<y && y<result->ny){
                                                index=y*result->nx+resultIndex_start_x;
                                                ResultImageType *resultPtr_XYZ = &resultPtr_Z[index];
                                                int *maskPtr_XYZ=&maskPtr_Z[index];
                                                for(int x=resultIndex_start_x; x<resultIndex_end_x; x++){
                                                    if(-1<x && x<result->nx){
                                                        ResultImageType value = *resultPtr_XYZ;
                                                        if(value==value && *maskPtr_XYZ>-1){
                                                            resultValues[resultIndex]=value;
                                                            resultOverlap[resultIndex]=1;
                                                        }
                                                    }
                                                    resultPtr_XYZ++;
                                                    resultIndex++;
                                                    maskPtr_XYZ++;
                                                }
                                            }
                                            else resultIndex+=BLOCK_WIDTH;
                                        }
                                    }
                                    else resultIndex+=BLOCK_WIDTH*BLOCK_WIDTH;
                                }
                                PrecisionTYPE targetMean=0.0;
                                PrecisionTYPE resultMean=0.0;
                                PrecisionTYPE voxelNumber=0.0;
                                for(int a=0; a<BLOCK_SIZE; a++){
                                    if(targetOverlap[a] && resultOverlap[a]){
                                        targetMean += (PrecisionTYPE)targetValues[a];
                                        resultMean += (PrecisionTYPE)resultValues[a];
                                        voxelNumber++;
                                    }
                                }

                                if(voxelNumber>BLOCK_SIZE/2){
                                    targetMean /= voxelNumber;
                                    resultMean /= voxelNumber;

                                    PrecisionTYPE targetVar=0.0;
                                    PrecisionTYPE resultVar=0.0;
                                    PrecisionTYPE localCC=0.0;

                                    for(int a=0; a<BLOCK_SIZE; a++){
                                        if(targetOverlap[a] && resultOverlap[a]){
                                            PrecisionTYPE targetTemp=(PrecisionTYPE)(targetValues[a]-targetMean);
                                            PrecisionTYPE resultTemp=(PrecisionTYPE)(resultValues[a]-resultMean);
                                            targetVar += (targetTemp)*(targetTemp);
                                            resultVar += (resultTemp)*(resultTemp);
                                            localCC += (targetTemp)*(resultTemp);
                                        }
                                    }

                                    localCC = fabs(localCC/sqrt(targetVar*resultVar));

                                    if(localCC>bestCC){
                                        bestCC=localCC;
                                        bestDisplacement[0] = (float)l;
                                        bestDisplacement[1] = (float)m;
                                        bestDisplacement[2] = (float)n;
                                    }
                                }
                            }
                        }
                    }
                    if(bestDisplacement[0]==bestDisplacement[0]){
                        float targetPosition_temp[3];
                        targetPosition_temp[0] = (float)(i*BLOCK_WIDTH);
                        targetPosition_temp[1] = (float)(j*BLOCK_WIDTH);
                        targetPosition_temp[2] = (float)(k*BLOCK_WIDTH);

                        bestDisplacement[0] += targetPosition_temp[0];
                        bestDisplacement[1] += targetPosition_temp[1];
                        bestDisplacement[2] += targetPosition_temp[2];

                        float tempPosition[3];
                        apply_affine(targetMatrix_xyz, targetPosition_temp, tempPosition);
                        params->targetPosition[activeBlockIndex] = tempPosition[0];
                        params->targetPosition[activeBlockIndex+1] = tempPosition[1];
                        params->targetPosition[activeBlockIndex+2] = tempPosition[2];
                        apply_affine(targetMatrix_xyz, bestDisplacement, tempPosition);
                        params->resultPosition[activeBlockIndex] = tempPosition[0];
                        params->resultPosition[activeBlockIndex+1] = tempPosition[1];
                        params->resultPosition[activeBlockIndex+2] = tempPosition[2];
                        activeBlockIndex += 3;
                        params->definedActiveBlock++;
                    }
                }
                blockIndex++;
            }
        }
    }
    free(resultValues);
    free(targetValues);
    free(targetOverlap);
    free(resultOverlap);
}
/* *************************************************************** */
// Called internally to determine the parameter type
template<typename PrecisionTYPE, typename TargetImageType>
        void block_matching_method2(   nifti_image * target,
                                       nifti_image * result,
                                       _reg_blockMatchingParam *params,
                                       int *mask)
{
    if(target->nz==1){
        switch(result->datatype){
        case NIFTI_TYPE_FLOAT64:
            block_matching_method2D<PrecisionTYPE, TargetImageType, double>
                    (target, result, params, mask);
            break;
        case NIFTI_TYPE_FLOAT32:
            block_matching_method2D<PrecisionTYPE, TargetImageType, float>
                    (target, result, params, mask);
            break;
        default:
            printf("[NiftyReg ERROR] block_match\tThe target image data type is not supported\n");
            return;
        }
    }
    else{
        switch(result->datatype){
            case NIFTI_TYPE_FLOAT64:
                block_matching_method3D<PrecisionTYPE, TargetImageType, double>
                        (target, result, params, mask);
                break;
            case NIFTI_TYPE_FLOAT32:
                block_matching_method3D<PrecisionTYPE, TargetImageType, float>
                        (target, result, params, mask);
                break;
            default:
                printf("[NiftyReg ERROR] block_match\tThe target image data type is not "
                       "supported\n");
                return;
        }
    }
}
/* *************************************************************** */
// Block matching interface function
template<typename PrecisionTYPE>
        void block_matching_method(	nifti_image * target,
                                        nifti_image * result,
                                        _reg_blockMatchingParam *params,
                                        int *mask)
{
    switch(target->datatype){
                case NIFTI_TYPE_FLOAT64:
        block_matching_method2<PrecisionTYPE, double>
                (target, result, params, mask);
        break;
            case NIFTI_TYPE_FLOAT32:
        block_matching_method2<PrecisionTYPE, float>
            (target, result, params, mask);
        break;
                default:
        printf("[NiftyReg ERROR] block_match\tThe target image data type is not"
               "supported\n");
        return;
    }
}
template void block_matching_method<float>(nifti_image *, nifti_image *, _reg_blockMatchingParam *, int *);
template void block_matching_method<double>(nifti_image *, nifti_image *, _reg_blockMatchingParam *, int *);
/* *************************************************************** */
/* *************************************************************** */
// Apply the suppled affine transformation to a 3D point
void apply_affine(mat44 * mat, float *pt, float *result)
{
    result[0] = (mat->m[0][0] * pt[0]) + (mat->m[0][1]*pt[1]) + (mat->m[0][2]*pt[2]) + (mat->m[0][3]);
    result[1] = (mat->m[1][0] * pt[0]) + (mat->m[1][1]*pt[1]) + (mat->m[1][2]*pt[2]) + (mat->m[1][3]);
    result[2] = (mat->m[2][0] * pt[0]) + (mat->m[2][1]*pt[1]) + (mat->m[2][2]*pt[2]) + (mat->m[2][3]);
}
/* *************************************************************** */
void apply_affine2D(mat44 * mat, float *pt, float *result)
{
    result[0] = (mat->m[0][0] * pt[0]) + (mat->m[0][1]*pt[1]) + (mat->m[0][3]);
    result[1] = (mat->m[1][0] * pt[0]) + (mat->m[1][1]*pt[1]) + (mat->m[1][3]);
}
/* *************************************************************** */
struct _reg_sorted_point3D
{
    float target[3];
    float result[3];

    double distance;

    _reg_sorted_point3D(float * t, float * r, double d)
            :distance(d)
    {
        target[0] = t[0];
        target[1] = t[1];
        target[2] = t[2];

        result[0] = r[0];
        result[1] = r[1];
        result[2] = r[2];
    }

    bool operator <(const _reg_sorted_point3D & sp) const
    {
        return (sp.distance < distance);
    }
};
/* *************************************************************** */
struct _reg_sorted_point2D
{
    float target[2];
    float result[2];

    double distance;

    _reg_sorted_point2D(float * t, float * r, double d)
            :distance(d)
    {
        target[0] = t[0];
        target[1] = t[1];

        result[0] = r[0];
        result[1] = r[1];
    }
    bool operator <(const _reg_sorted_point2D & sp) const
    {
        return (sp.distance < distance);
    }
};
/* *************************************************************** */
// Multiply matrices A and B together and store the result in r.
// We assume that the input pointers are valid and can store the result.
// A = ar * ac
// B = ac * bc
// r = ar * bc

// We can specify if we want to multiply A with the transpose of B

void mul_matrices(float ** a, float ** b, int ar, int ac, int bc, float ** r, bool transposeB)
{
    if (transposeB){
        for (int i = 0; i < ar; ++i){
            for (int j = 0; j < bc; ++j){
                r[i][j] = 0.0f;
                for (int k = 0; k < ac; ++k){
                    r[i][j] += a[i][k] * b[j][k];
                }
            }
        }
    }
    else{
        for (int i = 0; i < ar; ++i){
            for (int j = 0; j < bc; ++j){
                r[i][j] = 0.0f;
                for (int k = 0; k < ac; ++k){
                    r[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}
/* *************************************************************** */

// Multiply a matrix with a vctor
void mul_matvec(float ** a, int ar, int ac, float * b, float * r)
{
    for (int i = 0; i < ar; ++i){
        r[i] = 0;
        for (int k = 0; k < ac; ++k){
            r[i] += a[i][k] * b[k];
        }
    }
}
/* *************************************************************** */
// Compute determinant of a 3x3 matrix
float compute_determinant3x3(float ** mat)
{
    return 	(mat[0][0]*(mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1]))-
            (mat[0][1]*(mat[1][0]*mat[2][2]-mat[1][2]*mat[2][0]))+
            (mat[0][2]*(mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0]));
}
/* *************************************************************** */
// estimate an affine transformation using least square
void estimate_affine_transformation2D(std::vector<_reg_sorted_point2D> & points,
                                      mat44 * transformation,
                                      float ** A,
                                      float *  w,
                                      float ** v,
                                      float ** r,
                                      float *  b)
{
    int num_equations = points.size() * 2;
    unsigned c = 0;
    for (unsigned k = 0; k < points.size(); ++k){
        c = k * 2;
        A[c][0] = points[k].target[0];
        A[c][1] = points[k].target[1];
        A[c][2] = A[c][3] = A[c][5] = 0.0f;
        A[c][4] = 1.0f;

        A[c+1][2] = points[k].target[0];
        A[c+1][3] = points[k].target[1];
        A[c+1][0] = A[c+1][1] = A[c+1][4] = 0.0f;
        A[c+1][5] = 1.0f;
    }

    for (unsigned k = 0; k < 6; ++k){
        w[k] = 0.0f;
    }

    svd(A, num_equations, 6, w, v);

    for (unsigned k = 0; k < 6; ++k)
    {
        if (w[k] < 0.0001)
        {
            w[k] = 0.0f;
        }
        else{
            w[k] = 1.0f/w[k];
        }
    }

    // Now we can compute the pseudoinverse which is given by
    // V*inv(W)*U'
    // First compute the V * inv(w) in place.
    // Simply scale each column by the corresponding singular value
    for (unsigned k = 0; k < 6; ++k)
    {
        for (unsigned j = 0; j < 6; ++j){
            v[j][k] *=w[k];
        }
    }

    mul_matrices(v, A, 6, 6, num_equations, r, true);
    // Now r contains the pseudoinverse
    // Create vector b and then multiple rb to get the affine paramsA
    for (unsigned k = 0; k < points.size(); ++k){
        c = k * 2;
        b[c] = points[k].result[0];
        b[c+1] = points[k].result[1];
    }

    float * transform = new float[6];
    mul_matvec(r, 6, num_equations, b, transform);

    transformation->m[0][0] = transform[0];
    transformation->m[0][1] = transform[1];
    transformation->m[0][2] = 0.0f;
    transformation->m[0][3] = transform[4];

    transformation->m[1][0] = transform[2];
    transformation->m[1][1] = transform[3];
    transformation->m[1][2] = 0.0f;
    transformation->m[1][3] = transform[5];

    transformation->m[2][0] = 0.0f;
    transformation->m[2][1] = 0.0f;
    transformation->m[2][2] = 1.0f;
    transformation->m[2][3] = 0.0f;

    transformation->m[3][0] = 0.0f;
    transformation->m[3][1] = 0.0f;
    transformation->m[3][2] = 0.0f;
    transformation->m[3][3] = 1.0f;

    delete[] transform;
}



// estimate an affine transformation using least square
void estimate_affine_transformation3D(std::vector<_reg_sorted_point3D> & points,
                                      mat44 * transformation,
                                      float ** A,
                                      float *  w,
                                      float ** v,
                                      float ** r,
                                      float *  b)
{
    // Create our A matrix
    // we need at least 4 points. Assuming we have that here.
    int num_equations = points.size() * 3;
    unsigned c = 0;
    for (unsigned k = 0; k < points.size(); ++k)
    {
        c = k * 3;
        A[c][0] = points[k].target[0];
        A[c][1] = points[k].target[1];
        A[c][2] = points[k].target[2];
        A[c][3] = A[c][4] = A[c][5] = A[c][6] = A[c][7] = A[c][8] = A[c][10] = A[c][11] = 0.0f;
        A[c][9] = 1.0f;

        A[c+1][3] = points[k].target[0];
        A[c+1][4] = points[k].target[1];
        A[c+1][5] = points[k].target[2];
        A[c+1][0] = A[c+1][1] = A[c+1][2] = A[c+1][6] = A[c+1][7] = A[c+1][8] = A[c+1][9] = A[c+1][11] = 0.0f;
        A[c+1][10] = 1.0f;

        A[c+2][6] = points[k].target[0];
        A[c+2][7] = points[k].target[1];
        A[c+2][8] = points[k].target[2];
        A[c+2][0] = A[c+2][1] = A[c+2][2] = A[c+2][3] = A[c+2][4] = A[c+2][5] = A[c+2][9] = A[c+2][10] = 0.0f;
        A[c+2][11] = 1.0f;
    }

    for (unsigned k = 0; k < 12; ++k)
    {
        w[k] = 0.0f;
    }
    // Now we can compute our svd
    svd(A, num_equations, 12, w, v);

    // First we make sure that the really small singular values
    // are set to 0. and compute the inverse by taking the reciprocal
    // of the entries
    for (unsigned k = 0; k < 12; ++k)
    {
        if (w[k] < 0.0001)
        {
            w[k] = 0.0f;
        }
        else
        {
            w[k] = 1.0f/w[k];
        }
    }

    // Now we can compute the pseudoinverse which is given by
    // V*inv(W)*U'
    // First compute the V * inv(w) in place.
    // Simply scale each column by the corresponding singular value
    for (unsigned k = 0; k < 12; ++k)
    {
        for (unsigned j = 0; j < 12; ++j)
        {
            v[j][k] *=w[k];
        }
    }

    // Now multiply the matrices together
    // Pseudoinverse = v * e * A(transpose)
    mul_matrices(v, A, 12, 12, num_equations, r, true);
    // Now r contains the pseudoinverse
    // Create vector b and then multiple rb to get the affine paramsA
    for (unsigned k = 0; k < points.size(); ++k)
    {
        c = k * 3;
        b[c] = 		points[k].result[0];
        b[c+1] = 	points[k].result[1];
        b[c+2] = 	points[k].result[2];
    }

    float * transform = new float[12];
    mul_matvec(r, 12, num_equations, b, transform);

    transformation->m[0][0] = transform[0];
    transformation->m[0][1] = transform[1];
    transformation->m[0][2] = transform[2];
    transformation->m[0][3] = transform[9];

    transformation->m[1][0] = transform[3];
    transformation->m[1][1] = transform[4];
    transformation->m[1][2] = transform[5];
    transformation->m[1][3] = transform[10];

    transformation->m[2][0] = transform[6];
    transformation->m[2][1] = transform[7];
    transformation->m[2][2] = transform[8];
    transformation->m[2][3] = transform[11];

    transformation->m[3][0] = 0.0f;
    transformation->m[3][1] = 0.0f;
    transformation->m[3][2] = 0.0f;
    transformation->m[3][3] = 1.0f;

    delete[] transform;
}

void optimize_affine2D(_reg_blockMatchingParam * params,
                       mat44 * final)
{
    // Set the current transformation to identity
    final->m[0][0] = final->m[1][1] = final->m[2][2] = final->m[3][3] = 1.0f;
    final->m[0][1] = final->m[0][2] = final->m[0][3] = 0.0f;
    final->m[1][0] = final->m[1][2] = final->m[1][3] = 0.0f;
    final->m[2][0] = final->m[2][1] = final->m[2][3] = 0.0f;
    final->m[3][0] = final->m[3][1] = final->m[3][2] = 0.0f;

//    const unsigned num_points = params->activeBlockNumber;
    const unsigned num_points = params->definedActiveBlock;
    unsigned long num_equations = num_points * 2;
    std::multimap<double, _reg_sorted_point2D> queue;
    std::vector<_reg_sorted_point2D> top_points;
    double distance = 0.0;
    double lastDistance = std::numeric_limits<double>::max();;
    unsigned long i;

    // massive left hand side matrix
    float ** a = new float *[num_equations];
    for (unsigned k = 0; k < num_equations; ++k)
    {
        a[k] = new float[6]; // full affine
    }

    // The array of singular values returned by svd
    float *w = new float[6];

    // v will be n x n
    float **v = new float *[6];
    for (unsigned k = 0; k < 6; ++k)
    {
        v[k] = new float[6];
    }

    // Allocate memory for pseudoinverse
    float **r = new float *[6];
    for (unsigned k = 0; k < 6; ++k)
    {
        r[k] = new float[num_equations];
    }

    // Allocate memory for RHS vector
    float *b = new float[num_equations];

    // The initial vector with all the input points
    for (unsigned j = 0; j < num_points*2; j+=2)
    {
        top_points.push_back(_reg_sorted_point2D(&(params->targetPosition[j]),
                                                 &(params->resultPosition[j]),0.0f));
    }

    // estimate the optimal transformation while considering all the points
    estimate_affine_transformation2D(top_points, final, a, w, v, r, b);

    // Delete a, b and r. w and v will not change size in subsequent svd operations.
    for (unsigned int k = 0; k < num_equations; ++k)
    {
        delete[] a[k];
    }
    delete[] a;
    delete[] b;

    for (unsigned k = 0; k < 6; ++k)
    {
        delete[] r[k];
    }
    delete [] r;

    // The LS in the iterations is done on subsample of the input data
    float * newResultPosition = new float[num_points*2];
    const unsigned long num_to_keep = (unsigned long)(num_points * (params->percent_to_keep/100.0f));
    num_equations = num_to_keep*2;

    // The LHS matrix
    a = new float *[num_equations];
    for (unsigned k = 0; k < num_equations; ++k)
    {
        a[k] = new float[6]; // full affine
    }

    // Allocate memory for pseudoinverse
    r = new float *[6];
    for (unsigned k = 0; k < 6; ++k)
    {
        r[k] = new float[num_equations];
    }

    // Allocate memory for RHS vector
    b = new float[num_equations];
    mat44 lastTransformation;
    memset(&lastTransformation,0,sizeof(mat44));

    for (unsigned count = 0; count < MAX_ITERATIONS; ++count)
    {
        // Transform the points in the target
        for (unsigned j = 0; j < num_points * 2; j+=2)
        {
            apply_affine2D(final, &(params->targetPosition[j]), &newResultPosition[j]);
        }
        queue = std::multimap<double, _reg_sorted_point2D> ();
        for (unsigned j = 0; j < num_points * 2; j+=2)
        {
            distance = get_square_distance2D(&newResultPosition[j], &(params->resultPosition[j]));
            queue.insert(std::pair<double, _reg_sorted_point2D>(distance, _reg_sorted_point2D(&(params->targetPosition[j]),
                                           &(params->resultPosition[j]), distance)));
        }

        distance = 0.0;
        i = 0;
        top_points.clear();

        for (std::multimap<double, _reg_sorted_point2D>::iterator it = queue.begin();
        it != queue.end(); ++it, ++i)
        {
            if (i >= num_to_keep) break;
            top_points.push_back((*it).second);
            distance += (*it).first;
        }

        if ((distance > lastDistance) || (lastDistance - distance) < TOLERANCE)
        {
            // restore the last transformation
            copy_transformation_4x4(lastTransformation, *(final));
            break;
        }
        lastDistance = distance;
        copy_transformation_4x4(*(final), lastTransformation);
        estimate_affine_transformation2D(top_points, final, a, w, v, r, b);
    }

    delete[] newResultPosition;
    delete[] b;
    for (unsigned k = 0; k < 6; ++k)
    {
        delete[] r[k];
    }
    delete [] r;

    // free the memory
    for (unsigned int k = 0; k < num_equations; ++k)
    {
        delete[] a[k];
    }
    delete[] a;

    delete[] w;
    for (int k = 0; k < 6; ++k)
    {
        delete[] v[k];
    }
    delete [] v;
}

void optimize_affine3D(_reg_blockMatchingParam *params,
                        mat44 * final)
{
    // Set the current transformation to identity
    final->m[0][0] = final->m[1][1] = final->m[2][2] = final->m[3][3] = 1.0f;
    final->m[0][1] = final->m[0][2] = final->m[0][3] = 0.0f;
    final->m[1][0] = final->m[1][2] = final->m[1][3] = 0.0f;
    final->m[2][0] = final->m[2][1] = final->m[2][3] = 0.0f;
    final->m[3][0] = final->m[3][1] = final->m[3][2] = 0.0f;

//    const unsigned num_points = params->activeBlockNumber;
    const unsigned num_points = params->definedActiveBlock;
    unsigned long num_equations = num_points * 3;
    std::multimap<double, _reg_sorted_point3D> queue;
    std::vector<_reg_sorted_point3D> top_points;
    double distance = 0.0;
    double lastDistance = std::numeric_limits<double>::max();
    unsigned long i;

    // massive left hand side matrix
    float ** a = new float *[num_equations];
    for (unsigned k = 0; k < num_equations; ++k)
    {
        a[k] = new float[12]; // full affine
    }

    // The array of singular values returned by svd
    float *w = new float[12];

    // v will be n x n
    float **v = new float *[12];
    for (unsigned k = 0; k < 12; ++k)
    {
        v[k] = new float[12];
    }

    // Allocate memory for pseudoinverse
    float **r = new float *[12];
    for (unsigned k = 0; k < 12; ++k)
    {
        r[k] = new float[num_equations];
    }

    // Allocate memory for RHS vector
    float *b = new float[num_equations];

    // The initial vector with all the input points
    for (unsigned j = 0; j < num_points*3; j+=3)
    {
        top_points.push_back(_reg_sorted_point3D(&(params->targetPosition[j]),
                                                 &(params->resultPosition[j]),0.0f));
    }

    // estimate the optimal transformation while considering all the points
    estimate_affine_transformation3D(top_points, final, a, w, v, r, b);

    // Delete a, b and r. w and v will not change size in subsequent svd operations.
    for (unsigned int k = 0; k < num_equations; ++k)
    {
        delete[] a[k];
    }
    delete[] a;
    delete[] b;

    for (unsigned k = 0; k < 12; ++k)
    {
        delete[] r[k];
    }
    delete [] r;


    // The LS in the iterations is done on subsample of the input data
    float * newResultPosition = new float[num_points*3];
    const unsigned long num_to_keep = (unsigned long)(num_points * (params->percent_to_keep/100.0f));
    num_equations = num_to_keep*3;

    // The LHS matrix
    a = new float *[num_equations];
    for (unsigned k = 0; k < num_equations; ++k)
    {
        a[k] = new float[12]; // full affine
    }

    // Allocate memory for pseudoinverse
    r = new float *[12];
    for (unsigned k = 0; k < 12; ++k)
    {
        r[k] = new float[num_equations];
    }

    // Allocate memory for RHS vector
    b = new float[num_equations];
    mat44 lastTransformation;
    memset(&lastTransformation,0,sizeof(mat44));

    for (unsigned count = 0; count < MAX_ITERATIONS; ++count)
    {
        // Transform the points in the target
        for (unsigned j = 0; j < num_points * 3; j+=3)
        {
            apply_affine(final, &(params->targetPosition[j]), &newResultPosition[j]);
        }

        queue = std::multimap<double, _reg_sorted_point3D> ();
        for (unsigned j = 0; j < num_points * 3; j+=3)
        {
            distance = get_square_distance(&newResultPosition[j], &(params->resultPosition[j]));
            queue.insert(std::pair<double, _reg_sorted_point3D>(distance, _reg_sorted_point3D(&(params->targetPosition[j]),
                                           &(params->resultPosition[j]), distance)));
        }

        distance = 0.0;
        i = 0;
        top_points.clear();

        for (std::multimap<double, _reg_sorted_point3D>::iterator it = queue.begin();
        it != queue.end(); ++it, ++i)
        {
            if (i >= num_to_keep) break;
            top_points.push_back((*it).second);
            distance += (*it).first;
        }

        // If the change is not substantial or we are getting worst, we return
        if ((distance >= lastDistance) || (lastDistance - distance) < TOLERANCE)
        {
            // restore the last transformation
            copy_transformation_4x4(lastTransformation, *(final));
            break;
        }
        lastDistance = distance;
        copy_transformation_4x4(*(final), lastTransformation);
        estimate_affine_transformation3D(top_points, final, a, w, v, r, b);
    }
    delete[] newResultPosition;
    delete[] b;
    for (unsigned k = 0; k < 12; ++k)
    {
        delete[] r[k];
    }
    delete [] r;

    // free the memory
    for (unsigned int k = 0; k < num_equations; ++k)
    {
        delete[] a[k];
    }
    delete[] a;

    delete[] w;
    for (int k = 0; k < 12; ++k)
    {
        delete[] v[k];
    }
    delete [] v;
}
void estimate_rigid_transformation2D(  std::vector<_reg_sorted_point2D> & points,
                                       mat44 * transformation)
{
    float centroid_target[2] = {0.0f};
    float centroid_result[2] = {0.0f};

    for (unsigned j = 0; j < points.size(); ++j){
        centroid_target[0] += points[j].target[0];
        centroid_target[1] += points[j].target[1];
        centroid_result[0] += points[j].result[0];
        centroid_result[1] += points[j].result[1];
    }

    centroid_target[0] /= (float)(points.size());
    centroid_target[1] /= (float)(points.size());

    centroid_result[0] /= (float)(points.size());
    centroid_result[1] /= (float)(points.size());

    float ** u = new float*[2];
    float * w = new float[2];
    float ** v = new float*[2];
    float ** ut = new float*[2];
    float ** r = new float*[2];

    for (unsigned i = 0; i < 2; ++i){
        u[i] = new float[2];
        v[i] = new float[2];
        ut[i] = new float[2];
        r[i] = new float[2];
        w[i] = 0.0f;
        for (unsigned j = 0; j < 2; ++j){
            u[i][j] = v[i][j] = ut[i][j] = r[i][j] = 0.0f;
        }
    }

    // Demean the input points
    for (unsigned j = 0; j < points.size(); ++j){
        points[j].target[0] -= centroid_target[0];
        points[j].target[1] -= centroid_target[1];

        points[j].result[0] -= centroid_result[0];
        points[j].result[1] -= centroid_result[1];

        u[0][0] += points[j].target[0] * points[j].result[0];
        u[0][1] += points[j].target[0] * points[j].result[1];

        u[1][0] += points[j].target[1] * points[j].result[0];
        u[1][1] += points[j].target[1] * points[j].result[1];
    }

    svd(u, 2, 2, w, v);

    // Calculate transpose
    ut[0][0] = u[0][0];
    ut[1][0] = u[0][1];

    ut[0][1] = u[1][0];
    ut[1][1] = u[1][1];

    // Calculate the rotation matrix
    mul_matrices(v, ut, 2, 2, 2, r, false);

    float det = (r[0][0] * r[1][1]) - (r[0][1] * r[1][0]);

    // Take care of possible reflection
    if (det < 0.0f)
    {
        v[0][2] = -v[0][2];
        v[1][2] = -v[1][2];
        mul_matrices(v, ut, 2, 2, 2, r, false);
    }

    // Calculate the translation
    float t[2];
    t[0] = centroid_result[0] - (r[0][0] * centroid_target[0] +
                                 r[0][1] * centroid_target[1]);

    t[1] = centroid_result[1] - (r[1][0] * centroid_target[0] +
                                 r[1][1] * centroid_target[1]);

    transformation->m[0][0] = r[0][0];
    transformation->m[0][1] = r[0][1];
    transformation->m[0][3] = t[0];

    transformation->m[1][0] = r[1][0];
    transformation->m[1][1] = r[1][1];
    transformation->m[1][3] = t[1];

    transformation->m[2][0] = 0.0f;
    transformation->m[2][1] = 0.0f;
    transformation->m[2][2] = 1.0f;
    transformation->m[2][3] = 0.0f;

    transformation->m[0][2] = 0.0f;
    transformation->m[1][2] = 0.0f;
    transformation->m[3][2] = 0.0f;

    transformation->m[3][0] = 0.0f;
    transformation->m[3][1] = 0.0f;
    transformation->m[3][2] = 0.0f;
    transformation->m[3][3] = 1.0f;

    // Do the deletion here
    for (int i = 0; i < 2; ++i)
    {
        delete [] u[i];
        delete [] v[i];
        delete [] ut[i];
        delete [] r[i];
    }
    delete [] u;
    delete [] v;
    delete [] ut;
    delete [] r;
    delete [] w;
}
void estimate_rigid_transformation3D(std::vector<_reg_sorted_point3D> & points,
                                     mat44 * transformation)
{
    float centroid_target[3] = {0.0f};
    float centroid_result[3] = {0.0f};


    for (unsigned j = 0; j < points.size(); ++j)
    {
        centroid_target[0] += points[j].target[0];
        centroid_target[1] += points[j].target[1];
        centroid_target[2] += points[j].target[2];

        centroid_result[0] += points[j].result[0];
        centroid_result[1] += points[j].result[1];
        centroid_result[2] += points[j].result[2];
    }

    centroid_target[0] /= (float)(points.size());
    centroid_target[1] /= (float)(points.size());
    centroid_target[2] /= (float)(points.size());

    centroid_result[0] /= (float)(points.size());
    centroid_result[1] /= (float)(points.size());
    centroid_result[2] /= (float)(points.size());

    float ** u = new float*[3];
    float * w = new float[3];
    float ** v = new float*[3];
    float ** ut = new float*[3];
    float ** r = new float*[3];

    for (unsigned i = 0; i < 3; ++i)
    {
        u[i] = new float[3];
        v[i] = new float[3];
        ut[i] = new float[3];
        r[i] = new float[3];

        w[i] = 0.0f;


        for (unsigned j = 0; j < 3; ++j)
        {
            u[i][j] = v[i][j] = ut[i][j] = r[i][j] = 0.0f;
        }
    }

    // Demean the input points
    for (unsigned j = 0; j < points.size(); ++j)
    {
        points[j].target[0] -= centroid_target[0];
        points[j].target[1] -= centroid_target[1];
        points[j].target[2] -= centroid_target[2];

        points[j].result[0] -= centroid_result[0];
        points[j].result[1] -= centroid_result[1];
        points[j].result[2] -= centroid_result[2];

        u[0][0] += points[j].target[0] * points[j].result[0];
        u[0][1] += points[j].target[0] * points[j].result[1];
        u[0][2] += points[j].target[0] * points[j].result[2];

        u[1][0] += points[j].target[1] * points[j].result[0];
        u[1][1] += points[j].target[1] * points[j].result[1];
        u[1][2] += points[j].target[1] * points[j].result[2];

        u[2][0] += points[j].target[2] * points[j].result[0];
        u[2][1] += points[j].target[2] * points[j].result[1];
        u[2][2] += points[j].target[2] * points[j].result[2];

    }

    svd(u, 3, 3, w, v);

    // Calculate transpose
    ut[0][0] = u[0][0];
    ut[1][0] = u[0][1];
    ut[2][0] = u[0][2];

    ut[0][1] = u[1][0];
    ut[1][1] = u[1][1];
    ut[2][1] = u[1][2];

    ut[0][2] = u[2][0];
    ut[1][2] = u[2][1];
    ut[2][2] = u[2][2];

    // Calculate the rotation matrix
    mul_matrices(v, ut, 3, 3, 3, r, false);

    float det = compute_determinant3x3(r);

    // Take care of possible reflection
    if (det < 0.0f)
    {
        v[0][2] = -v[0][2];
        v[1][2] = -v[1][2];
        v[2][2] = -v[2][2];

    }
    // Calculate the rotation matrix
    mul_matrices(v, ut, 3, 3, 3, r, false);

    // Calculate the translation
    float t[3];
    t[0] = centroid_result[0] - (r[0][0] * centroid_target[0] +
                                 r[0][1] * centroid_target[1] +
                                 r[0][2] * centroid_target[2]);

    t[1] = centroid_result[1] - (r[1][0] * centroid_target[0] +
                                 r[1][1] * centroid_target[1] +
                                 r[1][2] * centroid_target[2]);

    t[2] = centroid_result[2] - (r[2][0] * centroid_target[0] +
                                 r[2][1] * centroid_target[1] +
                                 r[2][2] * centroid_target[2]);

    transformation->m[0][0] = r[0][0];
    transformation->m[0][1] = r[0][1];
    transformation->m[0][2] = r[0][2];
    transformation->m[0][3] = t[0];

    transformation->m[1][0] = r[1][0];
    transformation->m[1][1] = r[1][1];
    transformation->m[1][2] = r[1][2];
    transformation->m[1][3] = t[1];

    transformation->m[2][0] = r[2][0];
    transformation->m[2][1] = r[2][1];
    transformation->m[2][2] = r[2][2];
    transformation->m[2][3] = t[2];

    transformation->m[3][0] = 0.0f;
    transformation->m[3][1] = 0.0f;
    transformation->m[3][2] = 0.0f;
    transformation->m[3][3] = 1.0f;

    // Do the deletion here
    for (int i = 0; i < 3; ++i)
    {
        delete [] u[i];
        delete [] v[i];
        delete [] ut[i];
        delete [] r[i];
    }
    delete [] u;
    delete [] v;
    delete [] ut;
    delete [] r;
    delete [] w;
}


// Find the optimal rigid transformation that will
// bring the point clouds into alignment.
void optimize_rigid2D(  _reg_blockMatchingParam *params,
                        mat44 * final)
{
//    unsigned num_points = params->activeBlockNumber;
    const unsigned num_points = params->definedActiveBlock;
    // Keep a sorted list of the distance measure
    std::multimap<double, _reg_sorted_point2D> queue;

    std::vector<_reg_sorted_point2D> top_points;
    double distance = 0.0;
    double lastDistance = std::numeric_limits<double>::max();
    unsigned long i;

    // Set the current transformation to identity
    final->m[0][0] = final->m[1][1] = final->m[2][2] = final->m[3][3] = 1.0f;
    final->m[0][1] = final->m[0][2] = final->m[0][3] = 0.0f;
    final->m[1][0] = final->m[1][2] = final->m[1][3] = 0.0f;
    final->m[2][0] = final->m[2][1] = final->m[2][3] = 0.0f;
    final->m[3][0] = final->m[3][1] = final->m[3][2] = 0.0f;

    for (unsigned j = 0; j < num_points * 2; j+= 2){
        top_points.push_back(_reg_sorted_point2D(&(params->targetPosition[j]),
                                                 &(params->resultPosition[j]), 0.0f));
    }

    estimate_rigid_transformation2D(top_points, final);
    unsigned long num_to_keep = (unsigned long)(num_points * (params->percent_to_keep/100.0f));
    float * newResultPosition = new float[num_points*2];

    mat44 lastTransformation;
    memset(&lastTransformation,0,sizeof(mat44));

    for (unsigned count = 0; count < MAX_ITERATIONS; ++count){
        // Transform the points in the target
        for (unsigned j = 0; j < num_points * 2; j+=2){
            apply_affine2D(final, &(params->targetPosition[j]), &newResultPosition[j]);
        }
        queue = std::multimap<double, _reg_sorted_point2D>();
        for (unsigned j = 0; j < num_points * 2; j+= 2){
            distance = get_square_distance2D(&newResultPosition[j], &(params->resultPosition[j]));
            queue.insert(std::pair<double, _reg_sorted_point2D>(distance,
                        _reg_sorted_point2D(&(params->targetPosition[j]),
                                           &(params->resultPosition[j]), distance)));
        }

        distance = 0.0;
        i = 0;
        top_points.clear();
        for (std::multimap<double, _reg_sorted_point2D>::iterator it = queue.begin();
        it != queue.end(); ++it, ++i)
        {
            if (i >= num_to_keep) break;
            top_points.push_back((*it).second);
            distance += (*it).first;
        }

        // If the change is not substantial, we return
        if ((distance > lastDistance) || (lastDistance - distance) < TOLERANCE)
        {
            copy_transformation_4x4(lastTransformation, *(final));
            break;
        }
        lastDistance = distance;
        copy_transformation_4x4(*(final), lastTransformation);
        estimate_rigid_transformation2D(top_points, final);
    }
    delete [] newResultPosition;
}
void optimize_rigid3D(  _reg_blockMatchingParam *params,
                        mat44 * final)
{
//    const unsigned num_points = params->activeBlockNumber;
    const unsigned num_points = params->definedActiveBlock;
    // Keep a sorted list of the distance measure
    std::multimap<double, _reg_sorted_point3D> queue;
    std::vector<_reg_sorted_point3D> top_points;
    double distance = 0.0;
    double lastDistance = std::numeric_limits<double>::max();
    unsigned long i;

    // Set the current transformation to identity
    final->m[0][0] = final->m[1][1] = final->m[2][2] = final->m[3][3] = 1.0f;
    final->m[0][1] = final->m[0][2] = final->m[0][3] = 0.0f;
    final->m[1][0] = final->m[1][2] = final->m[1][3] = 0.0f;
    final->m[2][0] = final->m[2][1] = final->m[2][3] = 0.0f;
    final->m[3][0] = final->m[3][1] = final->m[3][2] = 0.0f;

    for (unsigned j = 0; j < num_points * 3; j+= 3){
        top_points.push_back(_reg_sorted_point3D(&(params->targetPosition[j]),
                                                 &(params->resultPosition[j]), 0.0f));
    }

    estimate_rigid_transformation3D(top_points, final);
    unsigned long num_to_keep = (unsigned long)(num_points * (params->percent_to_keep/100.0f));
    float * newResultPosition = new float[num_points*3];

    mat44 lastTransformation;
    memset(&lastTransformation,0,sizeof(mat44));

    for (unsigned count = 0; count < MAX_ITERATIONS; ++count){
        // Transform the points in the target
        for (unsigned j = 0; j < num_points * 3; j+=3){
            apply_affine(final, &(params->targetPosition[j]), &newResultPosition[j]);
        }
        queue = std::multimap<double, _reg_sorted_point3D>();
        for (unsigned j = 0; j < num_points * 3; j+= 3){
            distance = get_square_distance(&newResultPosition[j], &(params->resultPosition[j]));
            queue.insert(std::pair<double, _reg_sorted_point3D>(distance,
                                                                _reg_sorted_point3D(&(params->targetPosition[j]),
                                                                                    &(params->resultPosition[j]), distance)));
        }

        distance = 0.0;
        i = 0;
        top_points.clear();
        for (std::multimap<double, _reg_sorted_point3D>::iterator it = queue.begin();
        it != queue.end(); ++it, ++i)
        {
            if (i >= num_to_keep) break;
            top_points.push_back((*it).second);
            distance += (*it).first;
        }

        // If the change is not substantial, we return
        if ((distance > lastDistance) || (lastDistance - distance) < TOLERANCE)
        {
            copy_transformation_4x4(lastTransformation, *(final));
            break;
        }
        lastDistance = distance;
        copy_transformation_4x4(*(final), lastTransformation);
        estimate_rigid_transformation3D(top_points, final);
    }

    delete [] newResultPosition;
}


// Find the optimal affine transformation
void optimize(	_reg_blockMatchingParam *params,
                mat44 * final,
                bool affine)
{
    if(params->blockNumber[2]==1){
        if (affine){
            optimize_affine2D(params, final);
        }
        else{
            optimize_rigid2D(params, final);
        }
    }
    else{
        if (affine){
            optimize_affine3D(params, final);
        }
        else{
            optimize_rigid3D(params, final);
        }
    }
}
