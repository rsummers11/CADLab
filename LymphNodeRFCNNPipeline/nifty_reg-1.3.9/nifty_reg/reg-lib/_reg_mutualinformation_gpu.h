/*
 *  _reg_mutualinformation_gpu.h
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MUTUALINFORMATION_GPU_H
#define _REG_MUTUALINFORMATION_GPU_H

#include "_reg_blocksize_gpu.h"

extern "C++"
void reg_getEntropies2x2_gpu(nifti_image *targetImages,
                             nifti_image *resultImages,
                             //int type,
                             unsigned int *target_bins, // should be an array of size num_target_volumes
                             unsigned int *result_bins, // should be an array of size num_result_volumes
                             double *probaJointHistogram,
                             double *logJointHistogram,
                             float  **logJointHistogram_d,
                             double *entropies,
                             int *mask);

extern "C++"
void reg_getVoxelBasedNMIGradientUsingPW_gpu(nifti_image *targetImage,
                                            nifti_image *resultImage,
                                            cudaArray **targetImageArray_d,
                                            float **resultImageArray_d,
                                            float4 **resultGradientArray_d,
                                            float **logJointHistogram_d,
                                            float4 **voxelNMIGradientArray_d,
                                            int **targetMask_d,
                                            int activeVoxelNumber,
                                            double *entropies,
                                            int refBinning,
                                            int floBinning);

extern "C++"
void reg_getVoxelBasedNMIGradientUsingPW2x2_gpu(nifti_image *targetImage,
                                                nifti_image *resultImage,
                                                cudaArray **targetImageArray1_d,
                                                cudaArray **targetImageArray2_d,
                                                float **resultImageArray1_d,
                                                float **resultImageArray2_d,
                                                float4 **resultGradientArray1_d,
                                                float4 **resultGradientArray2_d,
                                                float **logJointHistogram_d,
                                                float4 **voxelNMIGradientArray_d,
                                                int **mask_d,
                                                int activeVoxelNumber,
                                                double *entropies,
                                                unsigned int *targetBinning,
                                                unsigned int *resultBinning);


#endif
