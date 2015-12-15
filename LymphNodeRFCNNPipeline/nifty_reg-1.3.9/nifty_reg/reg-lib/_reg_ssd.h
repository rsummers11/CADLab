/*
 *  _reg_ssd.h
 *  
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_SSD_H
#define _REG_SSD_H

#include "nifti1_io.h"

extern "C++"
double reg_getSSD(nifti_image *targetImage,
                  nifti_image *resultImage,
                  nifti_image *jacobianDeterminantImage,
                  int *mask
                  );

extern "C++"
void reg_getVoxelBasedSSDGradient(nifti_image *targetImage,
                                  nifti_image *resultImage,
                                  nifti_image *resultImageGradient,
                                  nifti_image *ssdGradientImage,
                                  nifti_image *jacobianDeterminantImage,
                                  float maxSD,
                                  int *mask
                                  );
#endif
