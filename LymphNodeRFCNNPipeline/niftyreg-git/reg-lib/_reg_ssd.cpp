/*
 *  _reg_ssd.cpp
 *
 *
 *  Created by Marc Modat on 19/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ssd.h"

//#define USE_LOG_SSD

/* *************************************************************** */
/* *************************************************************** */
reg_ssd::reg_ssd()
   : reg_measure()
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_ssd constructor called\n");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_ssd::InitialiseMeasure(nifti_image *refImgPtr,
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
   // Set the pointers using the parent class function
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

   // Check that the input images have the same number of time point
   if(this->referenceImagePointer->nt != this->floatingImagePointer->nt)
   {
      fprintf(stderr,"[NiftyReg ERROR] reg_nmi::InitialiseMeasure\n");
      fprintf(stderr,"[NiftyReg ERROR] This number of time point should\n");
      fprintf(stderr,"[NiftyReg ERROR] be the same for both input images\n");
      reg_exit(1);
   }
   // Input images are normalised between 0 and 1
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
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_ssd::InitialiseMeasure(). Active time point:");
   for(int i=0; i<this->referenceImagePointer->nt; ++i)
      if(this->activeTimePoint[i])
         printf(" %i",i);
   printf("\n");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
double reg_getSSDValue(nifti_image *referenceImage,
                       nifti_image *warpedImage,
                       bool *activeTimePoint,
                       nifti_image *jacobianDetImage,
                       int *mask,
                       float *currentValue
                      )
{

#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif
   // Create pointers to the reference and warped image data
   DTYPE *referencePtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *warpedPtr=static_cast<DTYPE *>(warpedImage->data);
   // Create a pointer to the Jacobian determinant image if defined
   DTYPE *jacDetPtr=NULL;
   if(jacobianDetImage!=NULL)
      jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);


   double SSD_global=0.0, n=0.0;
   double targetValue, resultValue, diff;

   // Loop over the different time points
   for(int time=0; time<referenceImage->nt; ++time)
   {
      if(activeTimePoint[time])
      {

         // Create pointers to the current time point of the reference and warped images
         DTYPE *currentRefPtr=&referencePtr[time*voxelNumber];
         DTYPE *currentWarPtr=&warpedPtr[time*voxelNumber];

         double SSD_local=0.;
         n=0.;
#if defined (_OPENMP)
         #pragma omp parallel for default(none) \
         shared(referenceImage, currentRefPtr, currentWarPtr, mask, \
                jacobianDetImage, jacDetPtr, voxelNumber) \
         private(voxel, targetValue, resultValue, diff) \
reduction(+:SSD_local) \
reduction(+:n)
#endif
         for(voxel=0; voxel<voxelNumber; ++voxel)
         {
            // Check if the current voxel belongs to the mask
            if(mask[voxel]>-1)
            {
               // Ensure that both ref and warped values are defined
               targetValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope +
                                      referenceImage->scl_inter);
               resultValue = (double)(currentWarPtr[voxel] * referenceImage->scl_slope +
                                      referenceImage->scl_inter);
               if(targetValue==targetValue && resultValue==resultValue)
               {
                  diff = reg_pow2(targetValue-resultValue);
//						if(diff>0) diff=log(diff);
                  // Jacobian determinant modulation of the ssd if required
                  if(jacDetPtr!=NULL)
                  {
                     SSD_local += diff * jacDetPtr[voxel];
                     n += jacDetPtr[voxel];
                  }
                  else
                  {
                     SSD_local += diff;
                     n += 1.0;
                  }
               }
            }
         }
         currentValue[time]=-SSD_local;
         SSD_global -= SSD_local/n;
      }
   }
   return SSD_global;
}
template double reg_getSSDValue<float>(nifti_image *,nifti_image *,bool *,nifti_image *,int *, float *);
template double reg_getSSDValue<double>(nifti_image *,nifti_image *,bool *,nifti_image *,int *, float *);
/* *************************************************************** */
double reg_ssd::GetSimilarityMeasureValue()
{
   // Check that all the specified image are of the same datatype
   if(this->warpedFloatingImagePointer->datatype != this->referenceImagePointer->datatype)
   {
      fprintf(stderr, "[NiftyReg ERROR] reg_ssd::GetSimilarityMeasureValue\n");
      fprintf(stderr, "[NiftyReg ERROR] Both input images are exepected to have the same type\n");
      reg_exit(1);
   }
   double SSDValue;
   switch(this->referenceImagePointer->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      SSDValue = reg_getSSDValue<float>
                 (this->referenceImagePointer,
                  this->warpedFloatingImagePointer,
                  this->activeTimePoint,
                  NULL, // HERE TODO this->forwardJacDetImagePointer,
                  this->referenceMaskPointer,
                  this->currentValue
                 );
      break;
   case NIFTI_TYPE_FLOAT64:
      SSDValue = reg_getSSDValue<double>
                 (this->referenceImagePointer,
                  this->warpedFloatingImagePointer,
                  this->activeTimePoint,
                  NULL, // HERE TODO this->forwardJacDetImagePointer,
                  this->referenceMaskPointer,
                  this->currentValue
                 );
      break;
   default:
      fprintf(stderr,"[NiftyReg ERROR] Result pixel type unsupported in the SSD computation function.\n");
      reg_exit(1);
   }

   // Backward computation
   if(this->isSymmetric)
   {
      // Check that all the specified image are of the same datatype
      if(this->warpedReferenceImagePointer->datatype != this->floatingImagePointer->datatype)
      {
         fprintf(stderr, "[NiftyReg ERROR] reg_nmi::GetSimilarityMeasureValue\n");
         fprintf(stderr, "[NiftyReg ERROR] Both input images are exepected to have the same type\n");
         reg_exit(1);
      }
      switch(this->floatingImagePointer->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         SSDValue += reg_getSSDValue<float>
                     (this->floatingImagePointer,
                      this->warpedReferenceImagePointer,
                      this->activeTimePoint,
                      NULL, // HERE TODO this->backwardJacDetImagePointer,
                      this->floatingMaskPointer,
                      this->currentValue
                     );
         break;
      case NIFTI_TYPE_FLOAT64:
         SSDValue += reg_getSSDValue<double>
                     (this->floatingImagePointer,
                      this->warpedReferenceImagePointer,
                      this->activeTimePoint,
                      NULL, // HERE TODO this->backwardJacDetImagePointer,
                      this->floatingMaskPointer,
                      this->currentValue
                     );
         break;
      default:
         fprintf(stderr,"[NiftyReg ERROR] Result pixel type unsupported in the SSD computation function.\n");
         reg_exit(1);
      }
   }
   return SSDValue;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_getVoxelBasedSSDGradient(nifti_image *referenceImage,
                                  nifti_image *warpedImage,
                                  bool *activeTimePoint,
                                  nifti_image *warpedImageGradient,
                                  nifti_image *ssdGradientImage,
                                  nifti_image *jacobianDetImage,
                                  int *mask)
{
   // Create pointers to the reference and warped images
#ifdef _WIN32
   long voxel;
   long voxelNumber = (long)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#else
   size_t voxel;
   size_t voxelNumber = (size_t)referenceImage->nx*referenceImage->ny*referenceImage->nz;
#endif

   DTYPE *refPtr=static_cast<DTYPE *>(referenceImage->data);
   DTYPE *warPtr=static_cast<DTYPE *>(warpedImage->data);

   // Pointer to the warped image gradient
   DTYPE *spatialGradPtr=static_cast<DTYPE *>(warpedImageGradient->data);

   // Create a pointer to the Jacobian determinant values if defined
   DTYPE *jacDetPtr=NULL;
   if(jacobianDetImage!=NULL)
      jacDetPtr=static_cast<DTYPE *>(jacobianDetImage->data);

   // Create an array to store the computed gradient per time point
   DTYPE *ssdGradPtrX=static_cast<DTYPE *>(ssdGradientImage->data);
   DTYPE *ssdGradPtrY = &ssdGradPtrX[voxelNumber];
   DTYPE *ssdGradPtrZ = NULL;
   if(referenceImage->nz>1)
      ssdGradPtrZ = &ssdGradPtrY[voxelNumber];

   double targetValue, resultValue, common;
   // Loop over the different time points
   for(int time=0; time<referenceImage->nt; ++time)
   {
      if(activeTimePoint[time])
      {
         // Create some pointers to the current time point image to be accessed
         DTYPE *currentRefPtr=&refPtr[time*voxelNumber];
         DTYPE *currentWarPtr=&warPtr[time*voxelNumber];

         // Create some pointers to the spatial gradient of the current warped volume
         DTYPE *spatialGradPtrX=&spatialGradPtr[time*warpedImageGradient->nu*voxelNumber];
         DTYPE *spatialGradPtrY=&spatialGradPtrX[voxelNumber];
         DTYPE *spatialGradPtrZ=NULL;
         if(referenceImage->nz>1)
            spatialGradPtrZ=&spatialGradPtrY[voxelNumber];

#if defined (_OPENMP)
         #pragma omp parallel for default(none) \
         shared(referenceImage, warpedImage, currentRefPtr, currentWarPtr, time, \
                mask, jacDetPtr, spatialGradPtrX, spatialGradPtrY, spatialGradPtrZ, \
                ssdGradPtrX, ssdGradPtrY, ssdGradPtrZ, voxelNumber) \
         private(voxel, targetValue, resultValue, common)
#endif
         for(voxel=0; voxel<voxelNumber; voxel++)
         {
            if(mask[voxel]>-1)
            {
               targetValue = (double)(currentRefPtr[voxel] * referenceImage->scl_slope +
                                      referenceImage->scl_inter);
               resultValue = (double)(currentWarPtr[voxel] * warpedImage->scl_slope +
                                      warpedImage->scl_inter);
               if(targetValue==targetValue && resultValue==resultValue)
               {
                  common = -2.0 * (targetValue - resultValue);
                  if(jacDetPtr!=NULL)
                     common *= jacDetPtr[voxel];

                  ssdGradPtrX[voxel] += (DTYPE)(common * spatialGradPtrX[voxel]);
                  ssdGradPtrY[voxel] += (DTYPE)(common * spatialGradPtrY[voxel]);

                  if(ssdGradPtrZ!=NULL)
                  {
                     ssdGradPtrZ[voxel] += (DTYPE)(common * spatialGradPtrZ[voxel]);
                  }
               }
            }
         }
      }
   }// loop over time points
}
/* *************************************************************** */
template void reg_getVoxelBasedSSDGradient<float>
(nifti_image *,nifti_image *,bool *,nifti_image *,nifti_image *,nifti_image *, int *);
template void reg_getVoxelBasedSSDGradient<double>
(nifti_image *,nifti_image *,bool *,nifti_image *,nifti_image *,nifti_image *, int *);
/* *************************************************************** */
void reg_ssd::GetVoxelBasedSimilarityMeasureGradient()
{
   // Check if all required input images are of the same data type
   int dtype = this->referenceImagePointer->datatype;
   if(this->warpedFloatingImagePointer->datatype != dtype ||
         this->warpedFloatingGradientImagePointer->datatype != dtype ||
         this->forwardVoxelBasedGradientImagePointer->datatype != dtype
     )
   {
      fprintf(stderr, "[NiftyReg ERROR] reg_nmi::GetVoxelBasedSimilarityMeasureGradient\n");
      fprintf(stderr, "[NiftyReg ERROR] Input images are exepected to be of the same type\n");
      reg_exit(1);
   }
   // Compute the gradient of the ssd for the forward transformation
   switch(dtype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_getVoxelBasedSSDGradient<float>
      (this->referenceImagePointer,
       this->warpedFloatingImagePointer,
       this->activeTimePoint,
       this->warpedFloatingGradientImagePointer,
       this->forwardVoxelBasedGradientImagePointer,
       NULL, // HERE TODO this->forwardJacDetImagePointer,
       this->referenceMaskPointer
      );
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_getVoxelBasedSSDGradient<double>
      (this->referenceImagePointer,
       this->warpedFloatingImagePointer,
       this->activeTimePoint,
       this->warpedFloatingGradientImagePointer,
       this->forwardVoxelBasedGradientImagePointer,
       NULL, // HERE TODO this->forwardJacDetImagePointer,
       this->referenceMaskPointer
      );
      break;
   default:
      fprintf(stderr,"[NiftyReg ERROR] reg_nmi::GetVoxelBasedSimilarityMeasureGradient\n");
      fprintf(stderr,"[NiftyReg ERROR] The input image data type is not supported\n");
      reg_exit(1);
   }
   // Compute the gradient of the ssd for the backward transformation
   if(this->isSymmetric)
   {
      dtype = this->floatingImagePointer->datatype;
      if(this->warpedReferenceImagePointer->datatype != dtype ||
            this->warpedReferenceGradientImagePointer->datatype != dtype ||
            this->backwardVoxelBasedGradientImagePointer->datatype != dtype
        )
      {
         fprintf(stderr, "[NiftyReg ERROR] reg_nmi::GetVoxelBasedSimilarityMeasureGradient\n");
         fprintf(stderr, "[NiftyReg ERROR] Input images are exepected to be of the same type\n");
         reg_exit(1);
      }
      // Compute the gradient of the nmi for the backward transformation
      switch(dtype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_getVoxelBasedSSDGradient<float>
         (this->floatingImagePointer,
          this->warpedReferenceImagePointer,
          this->activeTimePoint,
          this->warpedReferenceGradientImagePointer,
          this->backwardVoxelBasedGradientImagePointer,
          NULL, // HERE TODO this->backwardJacDetImagePointer,
          this->floatingMaskPointer
         );
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_getVoxelBasedSSDGradient<double>
         (this->floatingImagePointer,
          this->warpedReferenceImagePointer,
          this->activeTimePoint,
          this->warpedReferenceGradientImagePointer,
          this->backwardVoxelBasedGradientImagePointer,
          NULL, // HERE TODO this->backwardJacDetImagePointer,
          this->floatingMaskPointer
         );
         break;
      default:
         fprintf(stderr,"[NiftyReg ERROR] reg_nmi::GetVoxelBasedSimilarityMeasureGradient\n");
         fprintf(stderr,"[NiftyReg ERROR] The input image data type is not supported\n");
         reg_exit(1);
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
