/**
 * @file _reg_ssd.h
 * @brief File that contains sum squared difference related function
 * @author Marc Modat
 * @date 19/05/2009
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_DTI_H
#define _REG_DTI_H

//#include "_reg_measure.h"
#include "_reg_ssd.h" // HERE

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief DTI related measure of similarity class
class reg_dti : public reg_measure
{
public:
   /// @brief reg_dti class constructor
   reg_dti();
//    /// @brief Initialise the reg_dti object
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          int *maskFloPtr = NULL,
                          nifti_image *warRefImgPtr = NULL,
                          nifti_image *warRefGraPtr = NULL,
                          nifti_image *bckVoxBasedGraPtr = NULL);
//    /// @brief Returns the value
   virtual double GetSimilarityMeasureValue();
//    /// @brief Compute the voxel based gradient for DTI images
   virtual void GetVoxelBasedSimilarityMeasureGradient();
   /// @brief reg_dti class destructor
   ~reg_dti() {}
protected:
   // Store the indicies of the DT components in the order XX,XY,YY,XZ,YZ,ZZ
   unsigned int dtIndicies[6];
   float currentValue;
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

/** @brief Copmutes and returns the SSD between two input image
 * @param targetImage First input image to use to compute the metric
 * @param resultImage Second input image to use to compute the metric
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 * @return Returns an L2 measure of the distance between the anisotropic components of the diffusion tensors
 */
extern "C++" template <class DTYPE>
double reg_getDTIMeasureValue(nifti_image *targetImage,
                              nifti_image *resultImage,
                              int *mask,
                              unsigned int * dtIndicies
                             );

/** @brief Compute a voxel based gradient of the sum squared difference.
 * @param targetImage First input image to use to compute the metric
 * @param resultImage Second input image to use to compute the metric
 * @param resultImageGradient Spatial gradient of the input result image
 * @param dtiGradientImage Output image that will be updated with the
 * value of the dti measure gradient
 * @param maxSD Input scalar that contain the difference value between
 * the highest and the lowest intensity.
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 */
extern "C++" template <class DTYPE>
void reg_getVoxelBasedDTIMeasureGradient(nifti_image *referenceImage,
      nifti_image *warpedImage,
      nifti_image *warpedImageGradient,
      nifti_image *dtiMeasureGradientImage,
      int *mask,
      unsigned int * dtIndicies);
#endif
