/*
 *  _reg_resampling.h
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_H
#define _REG_RESAMPLING_H

#include "nifti1_io.h"
#include "_reg_globalTransformation.h"
#include "_reg_maths.h"

/** reg_resampleSourceImage
  * This function resample a source image into the space of a target/result image.
  * The deformation is provided by a 4D nifti image which is in the space of the target image.
  * In the 4D image, for each voxel i,j,k, the position in the real word for the source image is store.
  * Interpolation can be nearest Neighbor (0), linear (1) or cubic spline (3).
  * The cubic spline interpolation assume a padding value of 0
  * The padding value for the NN and the LIN interpolation are user defined.
 */
extern "C++"
void reg_resampleSourceImage(nifti_image *targetImage,
                             nifti_image *sourceImage,
                             nifti_image *resultImage,
                             nifti_image *positionField,
                             int *mask,
                             int interp,
                             float backgroundValue);

extern "C++"
void reg_getSourceImageGradient(nifti_image *targetImage,
                                nifti_image *sourceImage,
                                nifti_image *resultGradientImage,
                                nifti_image *deformationField,
                                int *mask,
                                int interp);

#endif
