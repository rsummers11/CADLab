/*
 *  _reg_affineTransformation.h
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_AFFINETRANSFORMATION_KERNELS_CU
#define _REG_AFFINETRANSFORMATION_KERNELS_CU

#include "_reg_common_gpu.h"

/* *************************************************************** */
/* *************************************************************** */
__device__ __constant__ int3 c_ImageSize;
__device__ __constant__ int c_VoxelNumber;
/* *************************************************************** */
texture<float4, 1, cudaReadModeElementType> txAffineTransformation;
/* *************************************************************** */
/* *************************************************************** */
__global__
void reg_affine_deformationField_kernel(float4 *PositionFieldArray)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_VoxelNumber){

        int3 imageSize = c_ImageSize;
        short3 voxelIndex;
        int tempIndex=tid;
        voxelIndex.z=(int)(tempIndex/((imageSize.x)*(imageSize.y)));
        tempIndex -= voxelIndex.z*(imageSize.x)*(imageSize.y);
        voxelIndex.y=(int)(tempIndex/(imageSize.x));
        voxelIndex.x = tempIndex - voxelIndex.y*(imageSize.x);

        /* The transformation is applied */
        float4 position;
        float4 matrix = tex1Dfetch(txAffineTransformation,0);
        position.x = 	matrix.x*voxelIndex.x + matrix.y*voxelIndex.y  +
                        matrix.z*voxelIndex.z  +  matrix.w;
        matrix = tex1Dfetch(txAffineTransformation,1);
        position.y = 	matrix.x*voxelIndex.x + matrix.y*voxelIndex.y  +
                        matrix.z*voxelIndex.z  +  matrix.w;
        matrix = tex1Dfetch(txAffineTransformation,2);
        position.z = 	matrix.x*voxelIndex.x + matrix.y*voxelIndex.y  +
                        matrix.z*voxelIndex.z  +  matrix.w;
        position.w=0.0f;
        /* the deformation field (real coordinates) is stored */
        PositionFieldArray[tid] = position;
    }
}
/* *************************************************************** */
/* *************************************************************** */

#endif
