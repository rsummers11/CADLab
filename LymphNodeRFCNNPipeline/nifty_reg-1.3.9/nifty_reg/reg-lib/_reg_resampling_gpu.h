/*
 *  _reg_resampling_gpu.h
 *  
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_GPU_H
#define _REG_RESAMPLING_GPU_H

#include "_reg_blocksize_gpu.h"

extern "C++"
void reg_resampleSourceImage_gpu(   nifti_image *sourceImage,
                                    float **resultImageArray_d,
                                    cudaArray **sourceImageArray_d,
                                    float4 **positionFieldImageArray_d,
                                    int **mask_d,
                                    int activeVoxelNumber,
                                    float sourceBGValue);

extern "C++"
void reg_getSourceImageGradient_gpu(nifti_image *sourceImage,
                                    cudaArray **sourceImageArray_d,
                                    float4 **positionFieldImageArray_d,
                                    float4 **resultGradientArray_d,
                                    int activeVoxelNumber);
#endif
