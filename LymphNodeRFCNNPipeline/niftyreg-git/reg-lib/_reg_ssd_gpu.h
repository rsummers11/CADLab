/*
 * @file _reg_ssd_gpu.h
 * @author Marc Modat
 * @date 14/11/2012
 *
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_SSD_GPU_H
#define _REG_SSD_GPU_H

#include "_reg_tools_gpu.h"
#include "_reg_measure_gpu.h"
#include "_reg_ssd.h"
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/// @brief SSD measure of similarity class on the device
class reg_ssd_gpu : public reg_ssd , public reg_measure_gpu
{
public:
   /// @brief reg_ssd class constructor
   reg_ssd_gpu();
   /// @brief Initialise the reg_ssd object
   virtual void InitialiseMeasure(nifti_image *refImgPtr,
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
   /// @brief Returns the ssd value
   double GetSimilarityMeasureValue();
   /// @brief Compute the voxel based ssd gradient
   void GetVoxelBasedSimilarityMeasureGradient();
   /// @brief Measure class desstructor
   ~reg_ssd_gpu() {}
};
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
extern "C++"
float reg_getSSDValue_gpu(nifti_image *referenceImage,
                          cudaArray **reference_d,
                          float **warped_d,
                          int **mask_d,
                          int activeVoxelNumber
                         );
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
extern "C++"
void reg_getVoxelBasedSSDGradient_gpu(nifti_image *referenceImage,
                                      cudaArray **reference_d,
                                      float **warped_d,
                                      float4 **spaGradient_d,
                                      float4 **ssdGradient_d,
                                      float maxSD,
                                      int **mask_d,
                                      int activeVoxelNumber
                                     );
#endif
