/*
 *  _reg_KLdivergence.h
 *
 *
 *  Created by Marc Modat on 14/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_KLDIV_H
#define _REG_KLDIV_H

#include "nifti1_io.h"
#include "_reg_maths.h"
#include <limits>

extern "C++"
double reg_getKLDivergence(nifti_image *reference,
                           nifti_image *warped,
                           nifti_image *jacobianDeterminantImage,
                           int *mask);

extern "C++"
void reg_getKLDivergenceVoxelBasedGradient(nifti_image *reference,
      nifti_image *warped,
      nifti_image *warpedGradient,
      nifti_image *KLdivGradient,
      nifti_image *jacobianDetImg,
      int *mask);
#endif
