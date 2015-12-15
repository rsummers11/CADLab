/**
 * @file  _reg_lncc.cpp
 * @author Aileen Corder
 * @author Marc Modat
 * @date 10/11/2012.
 * @brief CPP file for the LNCC related class and functions
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_LNCC_CPP
#define _REG_LNCC_CPP

#include "_reg_lncc.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_lncc::reg_lncc()
   : reg_measure()
{
   this->forwardCorrelationImage=NULL;
   this->referenceMeanImage=NULL;
   this->referenceSdevImage=NULL;
   this->warpedFloatingMeanImage=NULL;
   this->warpedFloatingSdevImage=NULL;

   this->backwardCorrelationImage=NULL;
   this->floatingMeanImage=NULL;
   this->floatingSdevImage=NULL;
   this->warpedReferenceMeanImage=NULL;
   this->warpedReferenceSdevImage=NULL;

   // Gaussian kernel is used by default
   this->kernelType=0;

   for(int i=0; i<255; ++i)
      kernelStandardDeviation[i]=-5.f;
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_lncc constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_lncc::~reg_lncc()
{
   if(this->forwardCorrelationImage!=NULL)
      nifti_image_free(this->forwardCorrelationImage);
   this->forwardCorrelationImage=NULL;
   if(this->referenceMeanImage!=NULL)
      nifti_image_free(this->referenceMeanImage);
   this->referenceMeanImage=NULL;
   if(this->referenceSdevImage!=NULL)
      nifti_image_free(this->referenceSdevImage);
   this->referenceSdevImage=NULL;
   if(this->warpedFloatingMeanImage!=NULL)
      nifti_image_free(this->warpedFloatingMeanImage);
   this->warpedFloatingMeanImage=NULL;
   if(this->warpedFloatingSdevImage!=NULL)
      nifti_image_free(this->warpedFloatingSdevImage);
   this->warpedFloatingSdevImage=NULL;

   if(this->backwardCorrelationImage!=NULL)
      nifti_image_free(this->backwardCorrelationImage);
   this->backwardCorrelationImage=NULL;
   if(this->floatingMeanImage!=NULL)
      nifti_image_free(this->floatingMeanImage);
   this->floatingMeanImage=NULL;
   if(this->floatingSdevImage!=NULL)
      nifti_image_free(this->floatingSdevImage);
   this->floatingSdevImage=NULL;
   if(this->warpedReferenceMeanImage!=NULL)
      nifti_image_free(this->warpedReferenceMeanImage);
   this->warpedReferenceMeanImage=NULL;
   if(this->warpedReferenceSdevImage!=NULL)
      nifti_image_free(this->warpedReferenceSdevImage);
   this->warpedReferenceSdevImage=NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void reg_lncc::UpdateLocalStatImages(nifti_image *originalImage,
                                     nifti_image *meanImage,
                                     nifti_image *stdDevImage,
                                     int *mask)
{
   DTYPE *origPtr = static_cast<DTYPE *>(originalImage->data);
   DTYPE *meanPtr = static_cast<DTYPE *>(meanImage->data);
   DTYPE *sdevPtr = static_cast<DTYPE *>(stdDevImage->data);
   memcpy(meanPtr, origPtr, originalImage->nvox*originalImage->nbyper);
   memcpy(sdevPtr, origPtr, originalImage->nvox*originalImage->nbyper);
   reg_tools_multiplyImageToImage(stdDevImage, stdDevImage, stdDevImage);
   reg_tools_kernelConvolution(meanImage, this->kernelStandardDeviation, this->kernelType, mask, this->activeTimePoint);
   reg_tools_kernelConvolution(stdDevImage, this->kernelStandardDeviation, this->kernelType, mask, this->activeTimePoint);

#ifdef _WIN32
   long voxel;
   long voxelNumber=(long)originalImage->nvox;
#else
   size_t voxel;
   size_t voxelNumber=originalImage->nvox;
#endif

#if defined (_OPENMP)
   #pragma omp parallel for default(none) \
   shared(voxelNumber, sdevPtr, meanPtr) \
   private(voxel)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      // G*(I^2) - (G*I)^2
      sdevPtr[voxel] = sqrt(sdevPtr[voxel] - reg_pow2(meanPtr[voxel]));
      // Stabilise the computation
      if(sdevPtr[voxel]<1.e-06) sdevPtr[voxel]=static_cast<DTYPE>(0);
   }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_lncc::InitialiseMeasure(nifti_image *refImgPtr,
                                 nifti_image *floImgPtr,
                                 int *maskRefPtr,
                                 nifti_image *warFloImgPtr,
                                 nifti_image *warFloGraPtr,
                                 nifti_image *forVoxBasedGraPtr,
                                 int *maskFloPtr,
                                 nifti_image *warRefImgPtr,
                                 nifti_image *warRefGraPtr,
                                 nifti_image *bckVoxBasedGraPtr)
{
   reg_measure::InitialiseMeasure(refImgPtr,
                                  floImgPtr,
                                  maskRefPtr,
                                  warFloImgPtr,
                                  warFloGraPtr,
                                  forVoxBasedGraPtr,
                                  maskFloPtr,
                                  warRefImgPtr,
                                  warRefGraPtr,
                                  bckVoxBasedGraPtr);

   for(int i=0; i<this->referenceImagePointer->nt; ++i)
   {
      if(this->activeTimePoint[i])
      {
         reg_intensityRescale(this->referenceImagePointer,
                              i,
                              0.f,
                              1.f);
         reg_intensityRescale(this->floatingImagePointer,
                              i,
                              0.f,
                              1.f);
      }
   }

   // Check that no images are already allocated
   if(this->forwardCorrelationImage!=NULL)
      nifti_image_free(this->forwardCorrelationImage);
   this->forwardCorrelationImage=NULL;
   if(this->referenceMeanImage!=NULL)
      nifti_image_free(this->referenceMeanImage);
   this->referenceMeanImage=NULL;
   if(this->referenceSdevImage!=NULL)
      nifti_image_free(this->referenceSdevImage);
   this->referenceSdevImage=NULL;
   if(this->warpedFloatingMeanImage!=NULL)
      nifti_image_free(this->warpedFloatingMeanImage);
   this->warpedFloatingMeanImage=NULL;
   if(this->warpedFloatingSdevImage!=NULL)
      nifti_image_free(this->warpedFloatingSdevImage);
   this->warpedFloatingSdevImage=NULL;
   if(this->backwardCorrelationImage!=NULL)
      nifti_image_free(this->backwardCorrelationImage);
   this->backwardCorrelationImage=NULL;
   if(this->floatingMeanImage!=NULL)
      nifti_image_free(this->floatingMeanImage);
   this->floatingMeanImage=NULL;
   if(this->floatingSdevImage!=NULL)
      nifti_image_free(this->floatingSdevImage);
   this->floatingSdevImage=NULL;
   if(this->warpedReferenceMeanImage!=NULL)
      nifti_image_free(this->warpedReferenceMeanImage);
   this->warpedReferenceMeanImage=NULL;
   if(this->warpedReferenceSdevImage!=NULL)
      nifti_image_free(this->warpedReferenceSdevImage);
   this->warpedReferenceSdevImage=NULL;
   // Allocate the required image to store mean and Sdev dev of the reference image
   this->forwardCorrelationImage=nifti_copy_nim_info(this->referenceImagePointer);
   this->forwardCorrelationImage->data=(void *)malloc(this->forwardCorrelationImage->nvox *
                                       this->forwardCorrelationImage->nbyper);

   this->referenceMeanImage=nifti_copy_nim_info(this->referenceImagePointer);
   this->referenceMeanImage->data=(void *)malloc(this->referenceMeanImage->nvox *
                                  this->referenceMeanImage->nbyper);

   this->referenceSdevImage=nifti_copy_nim_info(this->referenceImagePointer);
   this->referenceSdevImage->data=(void *)malloc(this->referenceSdevImage->nvox *
                                  this->referenceSdevImage->nbyper);

   this->warpedFloatingMeanImage=nifti_copy_nim_info(this->warpedFloatingImagePointer);
   this->warpedFloatingMeanImage->data=(void *)malloc(this->warpedFloatingMeanImage->nvox *
                                       this->warpedFloatingMeanImage->nbyper);

   this->warpedFloatingSdevImage=nifti_copy_nim_info(this->warpedFloatingImagePointer);
   this->warpedFloatingSdevImage->data=(void *)malloc(this->warpedFloatingSdevImage->nvox *
                                       this->warpedFloatingSdevImage->nbyper);
   // Compute local mean and reference image
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      this->UpdateLocalStatImages<float>(this->referenceImagePointer,
                                         this->referenceMeanImage,
                                         this->referenceSdevImage,
                                         this->referenceMaskPointer);
      break;
   case NIFTI_TYPE_FLOAT64:
      this->UpdateLocalStatImages<double>(this->referenceImagePointer,
                                          this->referenceMeanImage,
                                          this->referenceSdevImage,
                                          this->referenceMaskPointer);
      break;
   }
   if(this->isSymmetric)
   {
      this->backwardCorrelationImage=nifti_copy_nim_info(this->floatingImagePointer);
      this->backwardCorrelationImage->data=(void *)malloc(this->backwardCorrelationImage->nvox *
                                           this->backwardCorrelationImage->nbyper);

      this->floatingMeanImage=nifti_copy_nim_info(this->floatingImagePointer);
      this->floatingMeanImage->data=(void *)malloc(this->floatingMeanImage->nvox *
                                    this->floatingMeanImage->nbyper);

      this->floatingSdevImage=nifti_copy_nim_info(this->floatingImagePointer);
      this->floatingSdevImage->data=(void *)malloc(this->floatingSdevImage->nvox *
                                    this->floatingSdevImage->nbyper);

      this->warpedReferenceMeanImage=nifti_copy_nim_info(this->warpedReferenceImagePointer);
      this->warpedReferenceMeanImage->data=(void *)malloc(this->warpedReferenceMeanImage->nvox *
                                           this->warpedReferenceMeanImage->nbyper);

      this->warpedReferenceSdevImage=nifti_copy_nim_info(this->warpedReferenceImagePointer);
      this->warpedReferenceSdevImage->data=(void *)malloc(this->warpedReferenceSdevImage->nvox *
                                           this->warpedReferenceSdevImage->nbyper);
      // Compute local mean and floating image
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         this->UpdateLocalStatImages<float>(this->floatingImagePointer,
                                            this->floatingMeanImage,
                                            this->floatingSdevImage,
                                            this->floatingMaskPointer);
         break;
      case NIFTI_TYPE_FLOAT64:
         this->UpdateLocalStatImages<double>(this->floatingImagePointer,
                                             this->floatingMeanImage,
                                             this->floatingSdevImage,
                                             this->floatingMaskPointer);
         break;
      }
   }
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_lncc::InitialiseMeasure(). Active time point:");
   for(int i=0; i<this->referenceImagePointer->nt; ++i)
      if(this->activeTimePoint[i])
         printf(" %i",i);
   printf("\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class DTYPE>
double reg_getLNCCValue(nifti_image *referenceImage,
                        nifti_image *referenceMeanImage,
                        nifti_image *referenceSdevImage,
                        int *refMask,
                        nifti_image *warpedImage,
                        nifti_image *warpedMeanImage,
                        nifti_image *warpedSdevImage,
                        float *kernelStandardDeviation,
                        bool *activeTimePoint,
                        nifti_image *correlationImage,
                        int kernelType)
{
   // Compute the local correlation
   reg_tools_multiplyImageToImage(referenceImage, warpedImage, correlationImage);
   reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, refMask, activeTimePoint);

   double lncc_value_sum  = 0., lncc_value;
   double activeVoxel_num = 0.;

   DTYPE *refMeanPtr=static_cast<DTYPE *>(referenceMeanImage->data);
   DTYPE *warMeanPtr=static_cast<DTYPE *>(warpedMeanImage->data);
   DTYPE *refSdevPtr=static_cast<DTYPE *>(referenceSdevImage->data);
   DTYPE *warSdevPtr=static_cast<DTYPE *>(warpedSdevImage->data);
   DTYPE *correlaPtr=static_cast<DTYPE *>(correlationImage->data);
#ifdef _WIN32
   long voxel;
   long voxelNumber=(long)referenceImage->nx*
         referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber=(size_t)referenceImage->nx*
         referenceImage->ny*referenceImage->nz;
#endif

   // Iteration over all time points
   for(int t=0; t<referenceImage->nt; ++t)
   {
      if(activeTimePoint[t]==true)
      {
         DTYPE *refMeanPtr0 = &refMeanPtr[t*voxelNumber];
         DTYPE *warMeanPtr0 = &warMeanPtr[t*voxelNumber];
         DTYPE *refSdevPtr0 = &refSdevPtr[t*voxelNumber];
         DTYPE *warSdevPtr0 = &warSdevPtr[t*voxelNumber];
         DTYPE *correlaPtr0 = &correlaPtr[t*voxelNumber];
         // Iteration over all voxels
#if defined (_OPENMP)
         #pragma omp parallel for default(none) \
         shared(voxelNumber,refMask,refMeanPtr0,warMeanPtr0, \
                refSdevPtr0,warSdevPtr0,correlaPtr0) \
         private(voxel,lncc_value) \
reduction(+:lncc_value_sum) \
reduction(+:activeVoxel_num)
#endif
         for(voxel=0; voxel<voxelNumber; ++voxel)
         {
            // Check if the current voxel belongs to the mask
            if(refMask[voxel]>-1)
            {
               lncc_value = (
                               correlaPtr0[voxel] -
                               (refMeanPtr0[voxel]*warMeanPtr0[voxel])
                            ) /
                            (refSdevPtr0[voxel]*warSdevPtr0[voxel]);

               if(lncc_value==lncc_value && fabs(lncc_value)<1.01 && isinf(lncc_value)==0)
               {
                  lncc_value_sum += fabs(lncc_value);
                  ++activeVoxel_num;
               }
            }
         }
      }
   }
   return lncc_value_sum/activeVoxel_num;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_lncc::GetSimilarityMeasureValue()
{
   double lncc_value=0.f;
   // Update the local statistic images - Forward
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      this->UpdateLocalStatImages<float>(this->warpedFloatingImagePointer,
                                         this->warpedFloatingMeanImage,
                                         this->warpedFloatingSdevImage,
                                         this->referenceMaskPointer);
      break;
   case NIFTI_TYPE_FLOAT64:
      this->UpdateLocalStatImages<double>(this->warpedFloatingImagePointer,
                                          this->warpedFloatingMeanImage,
                                          this->warpedFloatingSdevImage,
                                          this->referenceMaskPointer);
      break;
   }
   // Compute the LNCC - Forward
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      lncc_value += reg_getLNCCValue<float>(this->referenceImagePointer,
                                            this->referenceMeanImage,
                                            this->referenceSdevImage,
                                            this->referenceMaskPointer,
                                            this->warpedFloatingImagePointer,
                                            this->warpedFloatingMeanImage,
                                            this->warpedFloatingSdevImage,
                                            this->kernelStandardDeviation,
                                            this->activeTimePoint,
                                            this->forwardCorrelationImage,
                                            this->kernelType);
      break;
   case NIFTI_TYPE_FLOAT64:
      lncc_value += reg_getLNCCValue<double>(this->referenceImagePointer,
                                             this->referenceMeanImage,
                                             this->referenceSdevImage,
                                             this->referenceMaskPointer,
                                             this->warpedFloatingImagePointer,
                                             this->warpedFloatingMeanImage,
                                             this->warpedFloatingSdevImage,
                                             this->kernelStandardDeviation,
                                             this->activeTimePoint,
                                             this->forwardCorrelationImage,
                                             this->kernelType);
      break;
   }
   if(this->isSymmetric)
   {
      // Update the local statistic images - Backward
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         this->UpdateLocalStatImages<float>(this->warpedReferenceImagePointer,
                                            this->warpedReferenceMeanImage,
                                            this->warpedReferenceSdevImage,
                                            this->floatingMaskPointer);
         break;
      case NIFTI_TYPE_FLOAT64:
         this->UpdateLocalStatImages<double>(this->warpedReferenceImagePointer,
                                             this->warpedReferenceMeanImage,
                                             this->warpedReferenceSdevImage,
                                             this->floatingMaskPointer);
         break;
      }
      // Compute the LNCC - Backward
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         lncc_value += reg_getLNCCValue<float>(this->floatingImagePointer,
                                               this->floatingMeanImage,
                                               this->floatingSdevImage,
                                               this->floatingMaskPointer,
                                               this->warpedReferenceImagePointer,
                                               this->warpedReferenceMeanImage,
                                               this->warpedReferenceSdevImage,
                                               this->kernelStandardDeviation,
                                               this->activeTimePoint,
                                               this->backwardCorrelationImage,
                                               this->kernelType);
         break;
      case NIFTI_TYPE_FLOAT64:
         lncc_value += reg_getLNCCValue<double>(this->floatingImagePointer,
                                                this->floatingMeanImage,
                                                this->floatingSdevImage,
                                                this->floatingMaskPointer,
                                                this->warpedReferenceImagePointer,
                                                this->warpedReferenceMeanImage,
                                                this->warpedReferenceSdevImage,
                                                this->kernelStandardDeviation,
                                                this->activeTimePoint,
                                                this->backwardCorrelationImage,
                                                this->kernelType);
         break;
      }
   }
   return lncc_value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void reg_getVoxelBasedLNCCGradient(nifti_image *referenceImage,
                                   nifti_image *referenceMeanImage,
                                   nifti_image *referenceSdevImage,
                                   int *refMask,
                                   nifti_image *warpedImage,
                                   nifti_image *warpedMeanImage,
                                   nifti_image *warpedSdevImage,
                                   float *kernelStandardDeviation,
                                   bool *activeTimePoint,
                                   nifti_image *correlationImage,
                                   nifti_image *warpedGradientImage,
                                   nifti_image *lnccGradientImage,
                                   int kernelType)
{
   // Compute the local correlation
   reg_tools_multiplyImageToImage(referenceImage, warpedImage, correlationImage);
   reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, refMask, activeTimePoint);

   DTYPE *refImagePtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *warImagePtr=static_cast<DTYPE *>(warpedImage->data);
   DTYPE *refMeanPtr=static_cast<DTYPE *>(referenceMeanImage->data);
   DTYPE *warMeanPtr=static_cast<DTYPE *>(warpedMeanImage->data);
   DTYPE *refSdevPtr=static_cast<DTYPE *>(referenceSdevImage->data);
   DTYPE *warSdevPtr=static_cast<DTYPE *>(warpedSdevImage->data);
   DTYPE *correlaPtr=static_cast<DTYPE *>(correlationImage->data);

#ifdef _WIN32
   long voxel;
   long voxelNumber=(long)referenceImage->nx*
         referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber=(size_t)referenceImage->nx*
         referenceImage->ny*referenceImage->nz;
#endif

   // Create some pointers to the gradient images
   DTYPE *lnccGradPtrX = static_cast<DTYPE *>(lnccGradientImage->data);
   DTYPE *warpGradPtrX = static_cast<DTYPE *>(warpedGradientImage->data);
   DTYPE *lnccGradPtrY = &lnccGradPtrX[voxelNumber];
   DTYPE *warpGradPtrY = &warpGradPtrX[voxelNumber];
   DTYPE *lnccGradPtrZ = NULL;
   DTYPE *warpGradPtrZ = NULL;
   if(referenceImage->nz>1)
   {
      lnccGradPtrZ = &lnccGradPtrY[voxelNumber];
      warpGradPtrZ = &warpGradPtrY[voxelNumber];
   }

   // Iteration over all time points to compute new values
   for(int t=0; t<referenceImage->nt; ++t)
   {
      DTYPE *refMeanPtr0 = &refMeanPtr[t*voxelNumber];
      DTYPE *warMeanPtr0 = &warMeanPtr[t*voxelNumber];
      DTYPE *refSdevPtr0 = &refSdevPtr[t*voxelNumber];
      DTYPE *warSdevPtr0 = &warSdevPtr[t*voxelNumber];
      DTYPE *correlaPtr0 = &correlaPtr[t*voxelNumber];
      double refMeanValue, warMeanValue, refSdevValue,
             warSdevValue, correlaValue;
      double temp1, temp2, temp3;
      // Iteration over all voxels
#if defined (_OPENMP)
      #pragma omp parallel for default(none) \
      shared(voxelNumber,refMask,refMeanPtr0,warMeanPtr0, \
             refSdevPtr0,warSdevPtr0,correlaPtr0) \
      private(voxel,refMeanValue,warMeanValue,refSdevValue, \
              warSdevValue, correlaValue, temp1, temp2, temp3)
#endif
      for(voxel=0; voxel<voxelNumber; ++voxel)
      {
         // Check if the current voxel belongs to the mask
         if(refMask[voxel]>-1)
         {

            refMeanValue = refMeanPtr0[voxel];
            warMeanValue = warMeanPtr0[voxel];
            refSdevValue = refSdevPtr0[voxel];
            warSdevValue = warSdevPtr0[voxel];
            correlaValue = correlaPtr0[voxel] - (refMeanValue*warMeanValue);

            temp1 = 1.0 / (refSdevValue * warSdevValue);
            temp2 = correlaValue /
                    (refSdevValue*warSdevValue*warSdevValue*warSdevValue);
            temp3 = (correlaValue * warMeanValue) /
                    (refSdevValue*warSdevValue*warSdevValue*warSdevValue)
                    -
                    refMeanValue / (refSdevValue * warSdevValue);
            if(temp1==temp1 && isinf(temp1)==0 &&
                  temp2==temp2 && isinf(temp2)==0 &&
                  temp3==temp3 && isinf(temp3)==0)
            {
               // Derivative of the absolute function
               if(correlaValue<0)
               {
                  temp1 *= -1.;
                  temp2 *= -1.;
                  temp3 *= -1.;
               }
               warMeanPtr0[voxel]=temp1;
               warSdevPtr0[voxel]=temp2;
               correlaPtr0[voxel]=temp3;
            }
            else warMeanPtr0[voxel]=warSdevPtr0[voxel]=correlaPtr0[voxel]=0.;
         }
         else warMeanPtr0[voxel]=warSdevPtr0[voxel]=correlaPtr0[voxel]=0.;
      }
   }
   // Smooth the newly computed values
   reg_tools_kernelConvolution(warpedMeanImage, kernelStandardDeviation, kernelType, refMask, activeTimePoint);
   reg_tools_kernelConvolution(warpedSdevImage, kernelStandardDeviation, kernelType, refMask, activeTimePoint);
   reg_tools_kernelConvolution(correlationImage, kernelStandardDeviation, kernelType, refMask, activeTimePoint);

   // Iteration over all time points to compute new values
   for(int t=0; t<referenceImage->nt; ++t)
   {
      // Pointers to the current reference and warped image time point
      DTYPE *temp1Ptr = &warMeanPtr[t*voxelNumber];
      DTYPE *temp2Ptr = &warSdevPtr[t*voxelNumber];
      DTYPE *temp3Ptr = &correlaPtr[t*voxelNumber];
      DTYPE *refImagePtr0 = &refImagePtr[t*voxelNumber];
      DTYPE *warImagePtr0 = &warImagePtr[t*voxelNumber];
      double common;
      // Iteration over all voxels
#if defined (_OPENMP)
      #pragma omp parallel for default(none) \
      shared(voxelNumber,refMask,refImagePtr0,warImagePtr0, \
             temp1Ptr,temp2Ptr,temp3Ptr,lnccGradPtrX,lnccGradPtrY,lnccGradPtrZ, \
             warpGradPtrX, warpGradPtrY, warpGradPtrZ) \
      private(voxel, common)
#endif
      for(voxel=0; voxel<voxelNumber; ++voxel)
      {
         // Check if the current voxel belongs to the mask
         if(refMask[voxel]>-1)
         {
            common = temp1Ptr[voxel] * refImagePtr0[voxel] -
                     temp2Ptr[voxel] * warImagePtr0[voxel] +
                     temp3Ptr[voxel];
            lnccGradPtrX[voxel] -= warpGradPtrX[voxel] * common;
            lnccGradPtrY[voxel] -= warpGradPtrY[voxel] * common;
            if(warpGradPtrZ!=NULL)
            {
               lnccGradPtrZ[voxel] -= warpGradPtrZ[voxel] * common;
            }
         }
      }
   }
   // Check for NaN
   DTYPE val;
#ifdef _WIN32
   voxelNumber=(long)lnccGradientImage->nvox;
#else
   voxelNumber=lnccGradientImage->nvox;
#endif
#if defined (_OPENMP)
   #pragma omp parallel for default(none) \
   shared(voxelNumber,lnccGradPtrX) \
   private(voxel, val)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      val=lnccGradPtrX[voxel];
      if(val!=val || isinf(val)!=0)
         lnccGradPtrX[voxel]=static_cast<DTYPE>(0);
   }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_lncc::GetVoxelBasedSimilarityMeasureGradient()
{
   // Update the local statistic images - Forward
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      this->UpdateLocalStatImages<float>(this->warpedFloatingImagePointer,
                                         this->warpedFloatingMeanImage,
                                         this->warpedFloatingSdevImage,
                                         this->referenceMaskPointer);
      break;
   case NIFTI_TYPE_FLOAT64:
      this->UpdateLocalStatImages<double>(this->warpedFloatingImagePointer,
                                          this->warpedFloatingMeanImage,
                                          this->warpedFloatingSdevImage,
                                          this->referenceMaskPointer);
      break;
   }
   // Compute the LNCC gradient - Forward
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_getVoxelBasedLNCCGradient<float>(this->referenceImagePointer,
                                           this->referenceMeanImage,
                                           this->referenceSdevImage,
                                           this->referenceMaskPointer,
                                           this->warpedFloatingImagePointer,
                                           this->warpedFloatingMeanImage,
                                           this->warpedFloatingSdevImage,
                                           this->kernelStandardDeviation,
                                           this->activeTimePoint,
                                           this->forwardCorrelationImage,
                                           this->warpedFloatingGradientImagePointer,
                                           this->forwardVoxelBasedGradientImagePointer,
                                           this->kernelType);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_getVoxelBasedLNCCGradient<double>(this->referenceImagePointer,
                                            this->referenceMeanImage,
                                            this->referenceSdevImage,
                                            this->referenceMaskPointer,
                                            this->warpedFloatingImagePointer,
                                            this->warpedFloatingMeanImage,
                                            this->warpedFloatingSdevImage,
                                            this->kernelStandardDeviation,
                                            this->activeTimePoint,
                                            this->forwardCorrelationImage,
                                            this->warpedFloatingGradientImagePointer,
                                            this->forwardVoxelBasedGradientImagePointer,
                                            this->kernelType);
      break;
   }
   if(this->isSymmetric)
   {
      // Update the local statistic images - Backward
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         this->UpdateLocalStatImages<float>(this->warpedReferenceImagePointer,
                                            this->warpedReferenceMeanImage,
                                            this->warpedReferenceSdevImage,
                                            this->floatingMaskPointer);
         break;
      case NIFTI_TYPE_FLOAT64:
         this->UpdateLocalStatImages<double>(this->warpedReferenceImagePointer,
                                             this->warpedReferenceMeanImage,
                                             this->warpedReferenceSdevImage,
                                             this->floatingMaskPointer);
         break;
      }
      // Compute the LNCC gradient - Backward
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getVoxelBasedLNCCGradient<float>(this->floatingImagePointer,
                                              this->floatingMeanImage,
                                              this->floatingSdevImage,
                                              this->floatingMaskPointer,
                                              this->warpedReferenceImagePointer,
                                              this->warpedReferenceMeanImage,
                                              this->warpedReferenceSdevImage,
                                              this->kernelStandardDeviation,
                                              this->activeTimePoint,
                                              this->backwardCorrelationImage,
                                              this->warpedReferenceGradientImagePointer,
                                              this->backwardVoxelBasedGradientImagePointer,
                                              this->kernelType);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getVoxelBasedLNCCGradient<double>(this->floatingImagePointer,
                                               this->floatingMeanImage,
                                               this->floatingSdevImage,
                                               this->floatingMaskPointer,
                                               this->warpedReferenceImagePointer,
                                               this->warpedReferenceMeanImage,
                                               this->warpedReferenceSdevImage,
                                               this->kernelStandardDeviation,
                                               this->activeTimePoint,
                                               this->backwardCorrelationImage,
                                               this->warpedReferenceGradientImagePointer,
                                               this->backwardVoxelBasedGradientImagePointer,
                                               this->kernelType);
         break;
      }
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif

