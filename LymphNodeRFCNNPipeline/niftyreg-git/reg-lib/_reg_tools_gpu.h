/*
 * @file _reg_tools_gpu.h
 * @author Marc Modat
 * @date 24/03/2009
 *
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_GPU_H
#define _REG_TOOLS_GPU_H

#include "_reg_common_gpu.h"
#include "_reg_tools.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

/* ******************************** */
/* ******************************** */
extern "C++"
void reg_voxelCentric2NodeCentric_gpu(nifti_image *targetImage,
                                      nifti_image *controlPointImage,
                                      float4 **voxelNMIGradientArray_d,
                                      float4 **nodeNMIGradientArray_d,
                                      float weight);
/* ******************************** */
/* ******************************** */
extern "C++"
void reg_convertNMIGradientFromVoxelToRealSpace_gpu(mat44 *sourceMatrix_xyz,
      nifti_image *controlPointImage,
      float4 **nodeNMIGradientArray_d);
/* ******************************** */
/* ******************************** */
extern "C++"
void reg_gaussianSmoothing_gpu( nifti_image *image,
                                float4 **imageArray_d,
                                float sigma,
                                bool axisToSmooth[8]);
/* ******************************** */
/* ******************************** */

extern "C++"
void reg_smoothImageForCubicSpline_gpu(nifti_image *resultImage,
                                       float4 **voxelNMIGradientArray_d,
                                       float *smoothingRadius);
/* ******************************** */
/* ******************************** */
extern "C++"
void reg_multiplyValue_gpu(int num, float4 **array_d, float value);
/* ******************************** */
/* ******************************** */
extern "C++"
void reg_addValue_gpu(int num, float4 **array_d, float value);
/* ******************************** */
/* ******************************** */
extern "C++"
void reg_multiplyArrays_gpu(int num, float4 **array1_d, float4 **array2_d);
/* ******************************** */
/* ******************************** */
extern "C++"
void reg_addArrays_gpu(int num, float4 **array1_d, float4 **array2_d);
/* ******************************** */
/* ******************************** */
extern "C++"
void reg_fillMaskArray_gpu(int num, int **array1_d);
/* ******************************** */
/* ******************************** */
extern "C++"
float reg_sumReduction_gpu(float *array_d,
                           int size);
/* ******************************** */
/* ******************************** */
extern "C++"
float reg_maxReduction_gpu(float *array_d,
                           int size);
/* ******************************** */
/* ******************************** */
extern "C++"
float reg_minReduction_gpu(float *array_d,
                           int size);
/* ******************************** */
/* ******************************** */

#endif

