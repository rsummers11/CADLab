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

#ifndef _REG_NMI_GPU_H
#define _REG_NMI_GPU_H

#include "_reg_nmi.h"
#include "_reg_measure_gpu.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief NMI measure of similarity class - GPU based
class reg_nmi_gpu : public reg_nmi , public reg_measure_gpu
{
public:
   /// @brief reg_nmi class constructor
   reg_nmi_gpu();
   /// @brief Initialise the reg_nmi_gpu object
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          int activeVoxNum,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          cudaArray **refDevicePtr,
                          cudaArray **floDevicePtr,
                          int **refMskDevicePtr,
                          float **warFloDevicePtr,
                          float4 **warFloGradDevicePtr,
                          float4 **forVoxBasedGraDevicePtr);
   /// @brief Returns the nmi value
   double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based nmi gradient
   void GetVoxelBasedSimilarityMeasureGradient();
   /// @brief reg_nmi class destructor
   ~reg_nmi_gpu();

protected:
   float *forwardJointHistogramLog_device;
//	float **backwardJointHistogramLog_device;
   void ClearHistogram();
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief NMI measure of similarity classe
class reg_multichannel_nmi_gpu : public reg_multichannel_nmi , public reg_measure_gpu
{
public:
   void InitialiseMeasure(nifti_image *refImgPtr,
                          nifti_image *floImgPtr,
                          int *maskRefPtr,
                          int activeVoxNum,
                          nifti_image *warFloImgPtr,
                          nifti_image *warFloGraPtr,
                          nifti_image *forVoxBasedGraPtr,
                          cudaArray **refDevicePtr,
                          cudaArray **floDevicePtr,
                          int **refMskDevicePtr,
                          float **warFloDevicePtr,
                          float4 **warFloGradDevicePtr,
                          float4 **forVoxBasedGraDevicePtr)
   {
      ;
   }
   /// @brief reg_nmi class constructor
   reg_multichannel_nmi_gpu() {}
   /// @brief Returns the nmi value
   double GetSimilarityMeasureValue()
   {
      return 0.;
   }
   /// @brief Compute the voxel based nmi gradient
   void GetVoxelBasedSimilarityMeasureGradient()
   {
      ;
   }
   /// @brief reg_nmi class destructor
   ~reg_multichannel_nmi_gpu() {}
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

extern "C++"
void reg_getVoxelBasedNMIGradient_gpu(nifti_image *referenceImage,
                                      cudaArray **referenceImageArray_d,
                                      float **warpedImageArray_d,
                                      float4 **resultGradientArray_d,
                                      float **logJointHistogram_d,
                                      float4 **voxelNMIGradientArray_d,
                                      int **targetMask_d,
                                      int activeVoxelNumber,
                                      double *entropies,
                                      int refBinning,
                                      int floBinning);

#endif
