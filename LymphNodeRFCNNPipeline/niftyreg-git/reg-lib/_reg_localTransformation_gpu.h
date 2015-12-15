/*
 *  _reg_spline_gpu.h
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_LOCALTRANSFORMATION_GPU_H
#define _REG_LOCALTRANSFORMATION_GPU_H

#include "_reg_common_gpu.h"
#include "_reg_maths.h"
#include "_reg_tools_gpu.h"
#include <limits>

extern "C++"
void reg_spline_getDeformationField_gpu(nifti_image *controlPointImage,
                                        nifti_image *targetImage,
                                        float4 **controlPointImageArray_d,
                                        float4 **positionFieldImageArray_d,
                                        int **mask,
                                        int activeVoxelNumber,
                                        bool bspline);

/* BE */
extern "C++"
float reg_spline_approxBendingEnergy_gpu(nifti_image *controlPointImage,
      float4 **controlPointImageArray_d);

extern "C++"
void reg_spline_approxBendingEnergyGradient_gpu(nifti_image *controlPointImage,
      float4 **controlPointImageArray_d,
      float4 **nodeGradientArray_d,
      float bendingEnergyWeight);

/** Jacobian
 *
 */
extern "C++"
double reg_spline_getJacobianPenaltyTerm_gpu(nifti_image *referenceImage,
      nifti_image *controlPointImage,
      float4 **controlPointImageArray_d,
      bool approx);

extern "C++"
void reg_spline_getJacobianPenaltyTermGradient_gpu(nifti_image *referenceImage,
      nifti_image *controlPointImage,
      float4 **controlPointImageArray_d,
      float4 **nodeGradientArray_d,
      float jacobianWeight,
      bool approx);

extern "C++"
double reg_spline_correctFolding_gpu(  nifti_image *targetImage,
                                       nifti_image *controlPointImage,
                                       float4 **controlPointImageArray_d,
                                       bool approx);

extern "C++"
void reg_getDeformationFieldFromVelocityGrid_gpu(nifti_image *cpp_h,
      nifti_image *def_h,
      float4 **cpp_gpu,
      float4 **def_gpu);

extern "C++"
void reg_defField_compose_gpu(nifti_image *def,
                              float4 **def_gpu,
                              float4 **defOut_gpu,
                              int **mask_gpu,
                              int activeVoxel);

extern "C++"
void reg_getDeformationFromDisplacement_gpu( nifti_image *image, float4 **imageArray_d);
extern "C++"
void reg_getDisplacementFromDeformation_gpu( nifti_image *image, float4 **imageArray_d);

extern "C++"
void reg_defField_getJacobianMatrix_gpu(nifti_image *deformationField,
                                        float4 **deformationField_gpu,
                                        float **jacobianMatrices_gpu);
#endif //_REG_LOCALTRANSFORMATION_GPU_H
