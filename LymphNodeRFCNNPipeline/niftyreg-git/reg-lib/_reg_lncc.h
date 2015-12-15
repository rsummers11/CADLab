/**
 * @file  _reg_lncc.h
 * @author Aileen Corder
 * @author Marc Modat
 * @date 10/11/2012.
 * @brief Header file for the LNCC related class and functions
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_LNCC_H
#define _REG_LNCC_H

#include "_reg_measure.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
class reg_lncc : public reg_measure
{
public:
   /// @brief reg_lncc class constructor
   reg_lncc();
   /// @brief reg_lncc class destructor
   ~reg_lncc();
   /// @brief Initialise the reg_lncc object
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
   /// @brief Returns the lncc value
   double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based lncc gradient
   void GetVoxelBasedSimilarityMeasureGradient();
   /// @brief Stuff
   void SetKernelStandardDeviation(int t, float stddev)
   {
      this->activeTimePoint[t]=true;
      this->kernelStandardDeviation[t]=stddev;
   }
   /// @brief Stuff
   void SetKernelType(int t)
   {
      this->kernelType=t;
   }
protected:
   float kernelStandardDeviation[255];
   nifti_image *forwardCorrelationImage;
   nifti_image *referenceMeanImage;
   nifti_image *referenceSdevImage;
   nifti_image *warpedFloatingMeanImage;
   nifti_image *warpedFloatingSdevImage;

   nifti_image *backwardCorrelationImage;
   nifti_image *floatingMeanImage;
   nifti_image *floatingSdevImage;
   nifti_image *warpedReferenceMeanImage;
   nifti_image *warpedReferenceSdevImage;

   int kernelType;

   template <class DTYPE>
   void UpdateLocalStatImages(nifti_image *imag,
                              nifti_image *mean,
                              nifti_image *sdev,
                              int *mask);
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @brief Copmutes and returns the LNCC between two input image
 * @param targetImage First input image to use to compute the metric
 * @param resultImage Second input image to use to compute the metric
 * @param gaussianStandardDeviation Standard deviation of the Gaussian kernel
 * to use.
 * @param mask Array that contains a mask to specify which voxel
 * should be considered. If set to NULL, all voxels are considered
 * @return Returns the computed LNCC
 */
extern "C++" template<class DTYPE>
double reg_getLNCCValue(nifti_image *referenceImage,
                        nifti_image *referenceMeanImage,
                        nifti_image *referenceStdDevImage,
                        int *mask,
                        nifti_image *warpedImage,
                        nifti_image *warpedMeanImage,
                        nifti_image *warpedStdDevImage,
                        float *kernelStdDev,
                        bool *activeTimePoint,
                        nifti_image *correlationImage,
                        int kernelType);

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/** @brief Compute a voxel based gradient of the LNCC.
 *  @param targetImage First input image to use to compute the metric
 *  @param resultImage Second input image to use to compute the metric
 *  @param resultImageGradient Spatial gradient of the input result image
 *  @param lnccGradientImage Output image that will be updated with the
 *  value of the LNCC gradient
 *  @param gaussianStandardDeviation Standard deviation of the Gaussian kernel
 *  to use.
 *  @param mask Array that contains a mask to specify which voxel
 *  should be considered. If set to NULL, all voxels are considered
 */
extern "C++" template <class DTYPE>
void reg_getVoxelBasedLNCCGradient(nifti_image *referenceImage,
                                   nifti_image *referenceMeanImage,
                                   nifti_image *referenceStdDevImage,
                                   int *refMask,
                                   nifti_image *warpedImage,
                                   nifti_image *warpedMeanImage,
                                   nifti_image *warpedStdDevImage,
                                   float *kernelStdDev,
                                   bool *activeTimePoint,
                                   nifti_image *correlationImage,
                                   nifti_image *warpedGradientImage,
                                   nifti_image *lnccGradientImage,
                                   int kernelType);
#endif

