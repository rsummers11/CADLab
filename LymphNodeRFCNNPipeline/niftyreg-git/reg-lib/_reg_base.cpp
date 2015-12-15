/**
 * @file _reg_base.cpp
 * @author Marc Modat
 * @date 15/11/2012
 *
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BASE_CPP
#define _REG_BASE_CPP

#include "_reg_base.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_base<T>::reg_base(int refTimePoint,int floTimePoint)
{
   this->optimiser=NULL;
   this->maxiterationNumber=300;
   this->optimiseX=true;
   this->optimiseY=true;
   this->optimiseZ=true;
   this->perturbationNumber=0;
   this->useConjGradient=true;
   this->useApproxGradient=false;

   this->measure_ssd=NULL;
   this->measure_kld=NULL;
   this->measure_dti=NULL;
   this->measure_lncc=NULL;
   this->measure_nmi=NULL;
   this->measure_multichannel_nmi=NULL;

   this->similarityWeight=0.; // is automatically set depending of the penalty term weights

   this->executableName=(char *)"NiftyReg BASE";
   this->referenceTimePoint=refTimePoint;
   this->floatingTimePoint=floTimePoint;
   this->inputReference=NULL; // pointer to external
   this->inputFloating=NULL; // pointer to external
   this->maskImage=NULL; // pointer to external
   this->affineTransformation=NULL;  // pointer to external
   this->referenceMask=NULL;
   this->referenceSmoothingSigma=0.;
   this->floatingSmoothingSigma=0.;
   this->referenceThresholdUp=new float[this->referenceTimePoint];
   this->referenceThresholdLow=new float[this->referenceTimePoint];
   this->floatingThresholdUp=new float[this->floatingTimePoint];
   this->floatingThresholdLow=new float[this->floatingTimePoint];
   for(int i=0; i<this->referenceTimePoint; i++)
   {
      this->referenceThresholdUp[i]=std::numeric_limits<T>::max();
      this->referenceThresholdLow[i]=-std::numeric_limits<T>::max();
   }
   for(int i=0; i<this->floatingTimePoint; i++)
   {
      this->floatingThresholdUp[i]=std::numeric_limits<T>::max();
      this->floatingThresholdLow[i]=-std::numeric_limits<T>::max();
   }
   this->warpedPaddingValue=std::numeric_limits<T>::quiet_NaN();
   this->levelNumber=3;
   this->levelToPerform=0;
   this->gradientSmoothingSigma=0;
   this->verbose=true;
   this->usePyramid=true;
   this->forwardJacobianMatrix=NULL;


   this->initialised=false;
   this->referencePyramid=NULL;
   this->floatingPyramid=NULL;
   this->maskPyramid=NULL;
   this->activeVoxelNumber=NULL;
   this->currentReference=NULL;
   this->currentFloating=NULL;
   this->currentMask=NULL;
   this->warped=NULL;
   this->deformationFieldImage=NULL;
   this->warpedGradientImage=NULL;
   this->voxelBasedMeasureGradientImage=NULL;

   this->interpolation=1;

   this->funcProgressCallback=NULL;
   this->paramsProgressCallback=NULL;

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::reg_base");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_base<T>::~reg_base()
{
   this->ClearWarped();
   this->ClearWarpedGradient();
   this->ClearDeformationField();
   this->ClearVoxelBasedMeasureGradient();
   if(this->referencePyramid!=NULL)
   {
      if(this->usePyramid)
      {
         for(unsigned int i=0; i<this->levelToPerform; i++)
         {
            if(referencePyramid[i]!=NULL)
            {
               nifti_image_free(referencePyramid[i]);
               referencePyramid[i]=NULL;
            }
         }
      }
      else
      {
         if(referencePyramid[0]!=NULL)
         {
            nifti_image_free(referencePyramid[0]);
            referencePyramid[0]=NULL;
         }
      }
      free(referencePyramid);
      referencePyramid=NULL;
   }
   if(this->maskPyramid!=NULL)
   {
      if(this->usePyramid)
      {
         for(unsigned int i=0; i<this->levelToPerform; i++)
         {
            if(this->maskPyramid[i]!=NULL)
            {
               free(this->maskPyramid[i]);
               this->maskPyramid[i]=NULL;
            }
         }
      }
      else
      {
         if(this->maskPyramid[0]!=NULL)
         {
            free(this->maskPyramid[0]);
            this->maskPyramid[0]=NULL;
         }
      }
      free(this->maskPyramid);
      maskPyramid=NULL;
   }
   if(this->floatingPyramid!=NULL)
   {
      if(this->usePyramid)
      {
         for(unsigned int i=0; i<this->levelToPerform; i++)
         {
            if(floatingPyramid[i]!=NULL)
            {
               nifti_image_free(floatingPyramid[i]);
               floatingPyramid[i]=NULL;
            }
         }
      }
      else
      {
         if(floatingPyramid[0]!=NULL)
         {
            nifti_image_free(floatingPyramid[0]);
            floatingPyramid[0]=NULL;
         }
      }
      free(floatingPyramid);
      floatingPyramid=NULL;
   }
   if(this->activeVoxelNumber!=NULL)
   {
      free(activeVoxelNumber);
      this->activeVoxelNumber=NULL;
   }
   if(this->referenceThresholdUp!=NULL)
   {
      delete []this->referenceThresholdUp;
      this->referenceThresholdUp=NULL;
   }
   if(this->referenceThresholdLow!=NULL)
   {
      delete []this->referenceThresholdLow;
      this->referenceThresholdLow=NULL;
   }
   if(this->floatingThresholdUp!=NULL)
   {
      delete []this->floatingThresholdUp;
      this->floatingThresholdUp=NULL;
   }
   if(this->floatingThresholdLow!=NULL)
   {
      delete []this->floatingThresholdLow;
      this->floatingThresholdLow=NULL;
   }
   if(this->activeVoxelNumber!=NULL)
   {
      delete []this->activeVoxelNumber;
      this->activeVoxelNumber=NULL;
   }
   if(this->optimiser!=NULL)
   {
      delete this->optimiser;
      this->optimiser=NULL;
   }

   if(this->measure_nmi!=NULL)
      delete this->measure_nmi;
   if(this->measure_multichannel_nmi!=NULL)
      delete this->measure_multichannel_nmi;
   if(this->measure_ssd!=NULL)
      delete this->measure_ssd;
   if(this->measure_kld!=NULL)
      delete this->measure_kld;
   if(this->measure_dti!=NULL)
      delete this->measure_dti;
   if(this->measure_lncc!=NULL)
      delete this->measure_lncc;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::~reg_base");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceImage(nifti_image *r)
{
   this->inputReference = r;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetReferenceImage");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingImage(nifti_image *f)
{
   this->inputFloating = f;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetFloatingImage");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetMaximalIterationNumber(unsigned int iter)
{
   this->maxiterationNumber=iter;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetMaximalIterationNumber");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceMask(nifti_image *m)
{
   this->maskImage = m;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetReferenceMask");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetAffineTransformation(mat44 *a)
{
   this->affineTransformation=a;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetAffineTransformation");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceSmoothingSigma(T s)
{
   this->referenceSmoothingSigma = s;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetReferenceSmoothingSigma");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingSmoothingSigma(T s)
{
   this->floatingSmoothingSigma = s;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetFloatingSmoothingSigma");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceThresholdUp(unsigned int i, T t)
{
   this->referenceThresholdUp[i] = t;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetReferenceThresholdUp");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetReferenceThresholdLow(unsigned int i, T t)
{
   this->referenceThresholdLow[i] = t;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetReferenceThresholdLow");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingThresholdUp(unsigned int i, T t)
{
   this->floatingThresholdUp[i] = t;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetFloatingThresholdUp");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetFloatingThresholdLow(unsigned int i, T t)
{
   this->floatingThresholdLow[i] = t;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetFloatingThresholdLow");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetWarpedPaddingValue(T p)
{
   this->warpedPaddingValue = p;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetWarpedPaddingValue");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetLevelNumber(unsigned int l)
{
   this->levelNumber = l;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetLevelNumber");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetLevelToPerform(unsigned int l)
{
   this->levelToPerform = l;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetLevelToPerform");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetGradientSmoothingSigma(T g)
{
   this->gradientSmoothingSigma = g;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetGradientSmoothingSigma");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseConjugateGradient()
{
   this->useConjGradient = true;
   this->useApproxGradient = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseConjugateGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUseConjugateGradient()
{
   this->useConjGradient = false;
   this->useApproxGradient = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::DoNotUseConjugateGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseApproximatedGradient()
{
   this->useConjGradient = false;
   this->useApproxGradient = true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseApproximatedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUseApproximatedGradient()
{
   this->useConjGradient = true;
   this->useApproxGradient = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::DoNotUseApproximatedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::PrintOutInformation()
{
   this->verbose = true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::PrintOutInformation");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotPrintOutInformation()
{
   this->verbose = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::DoNotPrintOutInformation");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::DoNotUsePyramidalApproach()
{
   this->usePyramid=false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::DoNotUsePyramidalApproach");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseNeareatNeighborInterpolation()
{
   this->interpolation=0;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseNeareatNeighborInterpolation");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseLinearInterpolation()
{
   this->interpolation=1;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseLinearInterpolation");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseCubicSplineInterpolation()
{
   this->interpolation=3;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseCubicSplineInterpolation");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearCurrentInputImage()
{
   this->currentReference=NULL;
   this->currentMask=NULL;
   this->currentFloating=NULL;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ClearCurrentInputImage");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::AllocateWarped()
{
   if(this->currentReference==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The reference image is not defined\n");
      reg_exit(1);
   }
   reg_base<T>::ClearWarped();
   this->warped = nifti_copy_nim_info(this->currentReference);
   this->warped->dim[0]=this->warped->ndim=this->currentFloating->ndim;
   this->warped->dim[4]=this->warped->nt=this->currentFloating->nt;
   this->warped->pixdim[4]=this->warped->dt=1.0;
   this->warped->nvox =
      (size_t)this->warped->nx *
      (size_t)this->warped->ny *
      (size_t)this->warped->nz *
      (size_t)this->warped->nt;
   this->warped->scl_slope=1.f;
   this->warped->scl_inter=0.f;
   this->warped->datatype = this->currentFloating->datatype;
   this->warped->nbyper = this->currentFloating->nbyper;
   this->warped->data = (void *)calloc(this->warped->nvox, this->warped->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::AllocateWarped");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearWarped()
{
   if(this->warped!=NULL)
      nifti_image_free(this->warped);
   this->warped=NULL;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ClearWarped");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::AllocateDeformationField()
{
   if(this->currentReference==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The reference image is not defined\n");
      reg_exit(1);
   }
   reg_base<T>::ClearDeformationField();
   this->deformationFieldImage = nifti_copy_nim_info(this->currentReference);
   this->deformationFieldImage->dim[0]=this->deformationFieldImage->ndim=5;
   this->deformationFieldImage->dim[1]=this->deformationFieldImage->nx=this->currentReference->nx;
   this->deformationFieldImage->dim[2]=this->deformationFieldImage->ny=this->currentReference->ny;
   this->deformationFieldImage->dim[3]=this->deformationFieldImage->nz=this->currentReference->nz;
   this->deformationFieldImage->dim[4]=this->deformationFieldImage->nt=1;
   this->deformationFieldImage->pixdim[4]=this->deformationFieldImage->dt=1.0;
   if(this->currentReference->nz==1)
      this->deformationFieldImage->dim[5]=this->deformationFieldImage->nu=2;
   else this->deformationFieldImage->dim[5]=this->deformationFieldImage->nu=3;
   this->deformationFieldImage->pixdim[5]=this->deformationFieldImage->du=1.0;
   this->deformationFieldImage->dim[6]=this->deformationFieldImage->nv=1;
   this->deformationFieldImage->pixdim[6]=this->deformationFieldImage->dv=1.0;
   this->deformationFieldImage->dim[7]=this->deformationFieldImage->nw=1;
   this->deformationFieldImage->pixdim[7]=this->deformationFieldImage->dw=1.0;
   this->deformationFieldImage->nvox =
      (size_t)this->deformationFieldImage->nx *
      (size_t)this->deformationFieldImage->ny *
      (size_t)this->deformationFieldImage->nz *
      (size_t)this->deformationFieldImage->nt *
      (size_t)this->deformationFieldImage->nu;
   this->deformationFieldImage->nbyper = sizeof(T);
   if(sizeof(T)==sizeof(float))
      this->deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
   else this->deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
   this->deformationFieldImage->data = (void *)calloc(this->deformationFieldImage->nvox,
                                       this->deformationFieldImage->nbyper);
   this->deformationFieldImage->intent_code=NIFTI_INTENT_VECTOR;
   memset(this->deformationFieldImage->intent_name, 0, 16);
   strcpy(this->deformationFieldImage->intent_name,"NREG_TRANS");
   this->deformationFieldImage->intent_p1=DEF_FIELD;
   this->deformationFieldImage->scl_slope=1.f;
   this->deformationFieldImage->scl_inter=0.f;

   if(this->measure_dti!=NULL)
      this->forwardJacobianMatrix=(mat33 *)malloc(
                                     this->deformationFieldImage->nx *
                                     this->deformationFieldImage->ny *
                                     this->deformationFieldImage->nz *
                                     sizeof(mat33));
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::AllocateDeformationField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearDeformationField()
{
   if(this->deformationFieldImage!=NULL)
   {
      nifti_image_free(this->deformationFieldImage);
      this->deformationFieldImage=NULL;
   }
   if(this->forwardJacobianMatrix!=NULL)
      free(this->forwardJacobianMatrix);
   this->forwardJacobianMatrix=NULL;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ClearDeformationField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::AllocateWarpedGradient()
{
   if(this->deformationFieldImage==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The deformation field image is not defined\n");
      reg_exit(1);
   }
   reg_base<T>::ClearWarpedGradient();
   this->warpedGradientImage = nifti_copy_nim_info(this->deformationFieldImage);
   this->warpedGradientImage->dim[0]=this->warpedGradientImage->ndim=5;
   this->warpedGradientImage->nt = this->warpedGradientImage->dim[4] = this->currentFloating->nt;
   this->warpedGradientImage->nvox =
      (size_t)this->warpedGradientImage->nx *
      (size_t)this->warpedGradientImage->ny *
      (size_t)this->warpedGradientImage->nz *
      (size_t)this->warpedGradientImage->nt *
      (size_t)this->warpedGradientImage->nu;
   this->warpedGradientImage->data = (void *)calloc(this->warpedGradientImage->nvox,
                                     this->warpedGradientImage->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::AllocateWarpedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearWarpedGradient()
{
   if(this->warpedGradientImage!=NULL)
   {
      nifti_image_free(this->warpedGradientImage);
      this->warpedGradientImage=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ClearWarpedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::AllocateVoxelBasedMeasureGradient()
{
   if(this->deformationFieldImage==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The deformation field image is not defined\n");
      reg_exit(1);
   }
   reg_base<T>::ClearVoxelBasedMeasureGradient();
   this->voxelBasedMeasureGradientImage = nifti_copy_nim_info(this->deformationFieldImage);
   this->voxelBasedMeasureGradientImage->data = (void *)calloc(this->voxelBasedMeasureGradientImage->nvox,
         this->voxelBasedMeasureGradientImage->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::AllocateVoxelBasedMeasureGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::ClearVoxelBasedMeasureGradient()
{
   if(this->voxelBasedMeasureGradientImage!=NULL)
   {
      nifti_image_free(this->voxelBasedMeasureGradientImage);
      this->voxelBasedMeasureGradientImage=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ClearVoxelBasedMeasureGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::CheckParameters()
{
   // CHECK THAT BOTH INPUT IMAGES ARE DEFINED
   if(this->inputReference==NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] No reference image has been defined.\n");
      reg_exit(1);
   }
   if(this->inputFloating==NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] No floating image has been defined.\n");
      reg_exit(1);
   }

   // CHECK THE MASK DIMENSION IF IT IS DEFINED
   if(this->maskImage!=NULL)
   {
      if(this->inputReference->nx != this->maskImage->nx ||
            this->inputReference->ny != this->maskImage->ny ||
            this->inputReference->nz != this->maskImage->nz )
      {
         printf("x: %i %i\n",this->inputReference->nx, this->maskImage->nx);
         printf("y: %i %i\n",this->inputReference->ny, this->maskImage->ny);
         printf("z: %i %i\n",this->inputReference->nz, this->maskImage->nz);
         fprintf(stderr,"[NiftyReg ERROR] The mask image has different x, y or z dimension than the reference image.\n");
         reg_exit(1);
      }
   }

   // CHECK THE NUMBER OF LEVEL TO PERFORM
   if(this->levelToPerform>0)
   {
      this->levelToPerform=this->levelToPerform<this->levelNumber?this->levelToPerform:this->levelNumber;
   }
   else this->levelToPerform=this->levelNumber;
   if(this->levelToPerform==0 || this->levelToPerform>this->levelNumber)
      this->levelToPerform=this->levelNumber;

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::CheckParameters");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::InitialiseSimilarity()
{
   // SET THE DEFAULT MEASURE OF SIMILARITY IF NONE HAS BEEN SET
   if(this->measure_nmi==NULL &&
         this->measure_ssd==NULL &&
         this->measure_dti==NULL &&
         this->measure_lncc==NULL &&
         this->measure_lncc==NULL)
   {
      this->measure_nmi=new reg_nmi;
      for(int i=0; i<this->inputReference->nt; ++i)
         this->measure_nmi->SetActiveTimepoint(i);
   }
   if(this->measure_nmi!=NULL)
      this->measure_nmi->InitialiseMeasure(this->currentReference,
                                           this->currentFloating,
                                           this->currentMask,
                                           this->warped,
                                           this->warpedGradientImage,
                                           this->voxelBasedMeasureGradientImage
                                          );

   if(this->measure_multichannel_nmi!=NULL)
      this->measure_multichannel_nmi->InitialiseMeasure(this->currentReference,
            this->currentFloating,
            this->currentMask,
            this->warped,
            this->warpedGradientImage,
            this->voxelBasedMeasureGradientImage
                                                       );

   if(this->measure_ssd!=NULL)
      this->measure_ssd->InitialiseMeasure(this->currentReference,
                                           this->currentFloating,
                                           this->currentMask,
                                           this->warped,
                                           this->warpedGradientImage,
                                           this->voxelBasedMeasureGradientImage
                                          );

   if(this->measure_kld!=NULL)
      this->measure_kld->InitialiseMeasure(this->currentReference,
                                           this->currentFloating,
                                           this->currentMask,
                                           this->warped,
                                           this->warpedGradientImage,
                                           this->voxelBasedMeasureGradientImage
                                          );

   if(this->measure_lncc!=NULL)
      this->measure_lncc->InitialiseMeasure(this->currentReference,
                                            this->currentFloating,
                                            this->currentMask,
                                            this->warped,
                                            this->warpedGradientImage,
                                            this->voxelBasedMeasureGradientImage
                                           );

   if(this->measure_dti!=NULL)
      this->measure_dti->InitialiseMeasure(this->currentReference,
                                           this->currentFloating,
                                           this->currentMask,
                                           this->warped,
                                           this->warpedGradientImage,
                                           this->voxelBasedMeasureGradientImage
                                          );

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::InitialiseSimilarity");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::Initialise()
{
   if(this->initialised) return;

   this->CheckParameters();

   // CREATE THE PYRAMIDE IMAGES
   if(this->usePyramid)
   {
      this->referencePyramid = (nifti_image **)malloc(this->levelToPerform*sizeof(nifti_image *));
      this->floatingPyramid = (nifti_image **)malloc(this->levelToPerform*sizeof(nifti_image *));
      this->maskPyramid = (int **)malloc(this->levelToPerform*sizeof(int *));
      this->activeVoxelNumber= (int *)malloc(this->levelToPerform*sizeof(int));
   }
   else
   {
      this->referencePyramid = (nifti_image **)malloc(sizeof(nifti_image *));
      this->floatingPyramid = (nifti_image **)malloc(sizeof(nifti_image *));
      this->maskPyramid = (int **)malloc(sizeof(int *));
      this->activeVoxelNumber= (int *)malloc(sizeof(int));
   }

   // FINEST LEVEL OF REGISTRATION
   if(this->usePyramid)
   {
      reg_createImagePyramid<T>(this->inputReference, this->referencePyramid, this->levelNumber, this->levelToPerform);
      reg_createImagePyramid<T>(this->inputFloating, this->floatingPyramid, this->levelNumber, this->levelToPerform);
      if (this->maskImage!=NULL)
         reg_createMaskPyramid<T>(this->maskImage, this->maskPyramid, this->levelNumber, this->levelToPerform, this->activeVoxelNumber);
      else
      {
         for(unsigned int l=0; l<this->levelToPerform; ++l)
         {
            this->activeVoxelNumber[l]=this->referencePyramid[l]->nx*this->referencePyramid[l]->ny*this->referencePyramid[l]->nz;
            this->maskPyramid[l]=(int *)calloc(activeVoxelNumber[l],sizeof(int));
         }
      }
   }
   else
   {
      reg_createImagePyramid<T>(this->inputReference, this->referencePyramid, 1, 1);
      reg_createImagePyramid<T>(this->inputFloating, this->floatingPyramid, 1, 1);
      if (this->maskImage!=NULL)
         reg_createMaskPyramid<T>(this->maskImage, this->maskPyramid, 1, 1, this->activeVoxelNumber);
      else
      {
         this->activeVoxelNumber[0]=this->referencePyramid[0]->nx*this->referencePyramid[0]->ny*this->referencePyramid[0]->nz;
         this->maskPyramid[0]=(int *)calloc(activeVoxelNumber[0],sizeof(int));
      }
   }

   unsigned int pyramidalLevelNumber=1;
   if(this->usePyramid) pyramidalLevelNumber=this->levelToPerform;

   // SMOOTH THE INPUT IMAGES IF REQUIRED
   for(unsigned int l=0; l<this->levelToPerform; l++)
   {
      if(this->referenceSmoothingSigma!=0.0)
      {
         bool *active = new bool[this->referencePyramid[l]->nt];
         float *sigma = new float[this->referencePyramid[l]->nt];
         active[0]=true;
         for(int i=1; i<this->referencePyramid[l]->nt; ++i)
            active[i]=false;
         sigma[0]=this->referenceSmoothingSigma;
         reg_tools_kernelConvolution(this->referencePyramid[l], sigma, 0, NULL, active);
         delete []active;
         delete []sigma;
      }
      if(this->floatingSmoothingSigma!=0.0)
      {
         // Only the first image is smoothed
         bool *active = new bool[this->floatingPyramid[l]->nt];
         float *sigma = new float[this->floatingPyramid[l]->nt];
         active[0]=true;
         for(int i=1; i<this->floatingPyramid[l]->nt; ++i)
            active[i]=false;
         sigma[0]=this->floatingSmoothingSigma;
         reg_tools_kernelConvolution(this->floatingPyramid[l], sigma, 0, NULL, active);
         delete []active;
         delete []sigma;
      }
   }

   // THRESHOLD THE INPUT IMAGES IF REQUIRED
   for(unsigned int l=0; l<pyramidalLevelNumber; l++)
   {
      reg_thresholdImage<T>(this->referencePyramid[l],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
      reg_thresholdImage<T>(this->floatingPyramid[l],this->referenceThresholdLow[0], this->referenceThresholdUp[0]);
   }

   this->initialised=true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::Initialise");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::SetOptimiser()
{
   if(this->useConjGradient)
      this->optimiser=new reg_conjugateGradient<T>();
   else this->optimiser=new reg_optimiser<T>();
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetOptimiser");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_base<T>::ComputeSimilarityMeasure()
{
   double measure=0.;
   if(this->measure_nmi!=NULL)
      measure += this->measure_nmi->GetSimilarityMeasureValue();

   if(this->measure_multichannel_nmi!=NULL)
      measure += this->measure_multichannel_nmi->GetSimilarityMeasureValue();

   if(this->measure_ssd!=NULL)
      measure += this->measure_ssd->GetSimilarityMeasureValue();

   if(this->measure_kld!=NULL)
      measure += this->measure_kld->GetSimilarityMeasureValue();

   if(this->measure_lncc!=NULL)
      measure += this->measure_lncc->GetSimilarityMeasureValue();

   if(this->measure_dti!=NULL)
      measure += this->measure_dti->GetSimilarityMeasureValue();

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::ComputeSimilarityMeasure");
#endif
   return double(this->similarityWeight) * measure;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::GetVoxelBasedGradient()
{
   // The intensity gradient is first computed
//    if(this->measure_dti!=NULL){
//        reg_getImageGradient(this->currentFloating,
//                             this->warpedGradientImage,
//                             this->deformationFieldImage,
//                             this->currentMask,
//                             this->interpolation,
//                             this->warpedPaddingValue,
//                             this->measure_dti->GetActiveTimepoints(),
//		 					   this->forwardJacobianMatrix,
//							   this->warped);
//    }
//    else{
   reg_getImageGradient(this->currentFloating,
                        this->warpedGradientImage,
                        this->deformationFieldImage,
                        this->currentMask,
                        this->interpolation,
                        this->warpedPaddingValue);
//    }

   // The voxel based gradient image is filled with zeros
   reg_tools_multiplyValueToImage(this->voxelBasedMeasureGradientImage,
                                  this->voxelBasedMeasureGradientImage,
                                  0.f);
   // The gradient of the various measures of similarity are computed
   if(this->measure_nmi!=NULL)
      this->measure_nmi->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_multichannel_nmi!=NULL)
      this->measure_multichannel_nmi->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_ssd!=NULL)
      this->measure_ssd->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_kld!=NULL)
      this->measure_kld->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_lncc!=NULL)
      this->measure_lncc->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_dti!=NULL)
      this->measure_dti->GetVoxelBasedSimilarityMeasureGradient();

#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::GetVoxelBasedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
//template<class T>
//void reg_base<T>::ApproximateParzenWindow()
//{
//    if(this->measure_nmi==NULL)
//        this->measure_nmi=new reg_nmi;
//    this->measure_nmi=approxParzenWindow = true;
//    return;
//}
///* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
//template<class T>
//void reg_base<T>::DoNotApproximateParzenWindow()
//{
//    if(this->measure_nmi==NULL)
//        this->measure_nmi=new reg_nmi;
//    this->measure_nmi=approxParzenWindow = false;
//    return;
//}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseNMISetReferenceBinNumber(int timepoint, int refBinNumber)
{
   if(this->measure_nmi==NULL)
      this->measure_nmi=new reg_nmi;
   this->measure_nmi->SetActiveTimepoint(timepoint);
   // I am here adding 4 to the specified bin number to accomodate for
   // the spline support
   this->measure_nmi->SetReferenceBinNumber(refBinNumber+4, timepoint);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseNMISetReferenceBinNumber");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseNMISetFloatingBinNumber(int timepoint, int floBinNumber)
{
   if(this->measure_nmi==NULL)
      this->measure_nmi=new reg_nmi;
   this->measure_nmi->SetActiveTimepoint(timepoint);
   // I am here adding 4 to the specified bin number to accomodate for
   // the spline support
   this->measure_nmi->SetFloatingBinNumber(floBinNumber+4, timepoint);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseNMISetFloatingBinNumber");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseMultiChannelNMI(int timepointNumber)
{
   if(this->measure_multichannel_nmi==NULL)
      this->measure_multichannel_nmi=new reg_multichannel_nmi;
   for(int i=0; i<timepointNumber; ++i)
      this->measure_multichannel_nmi->SetActiveTimepoint(i);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseMultiChannelNMI");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseSSD(int timepoint)
{
   if(this->measure_ssd==NULL)
      this->measure_ssd=new reg_ssd;
   this->measure_ssd->SetActiveTimepoint(timepoint);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseSSD");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseKLDivergence(int timepoint)
{
   if(this->measure_kld==NULL)
      this->measure_kld=new reg_kld;
   this->measure_kld->SetActiveTimepoint(timepoint);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseKLDivergence");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseLNCC(int timepoint, float stddev)
{
   if(this->measure_lncc==NULL)
      this->measure_lncc=new reg_lncc;
   this->measure_lncc->SetKernelStandardDeviation(timepoint,
         stddev);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseLNCC");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::SetLNCCKernelType(int type)
{
   if(this->measure_lncc==NULL)
   {
      reg_print_fct_error("reg_base<T>::SetLNCCKernelType");
      reg_print_msg_error("The LNCC object has to be created first");
      reg_exit(1);
   }
   this->measure_lncc->SetKernelType(type);
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::SetLNCCKernelType");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_base<T>::UseDTI(bool *timepoint)
{
   if(this->measure_dti==NULL)
      this->measure_dti=new reg_dti;
   for(int i=0; i<this->inputReference->nt; ++i)
   {
      if(timepoint[i]==true)
         this->measure_dti->SetActiveTimepoint(i);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::UseDTI");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::WarpFloatingImage(int inter)
{
   // Compute the deformation field
   this->GetDeformationField();

   if(this->measure_dti==NULL)
   {
      // Resample the floating image
      reg_resampleImage(this->currentFloating,
                        this->warped,
                        this->deformationFieldImage,
                        this->currentMask,
                        inter,
                        this->warpedPaddingValue);
   }
   else
   {
      reg_defField_getJacobianMatrix(this->deformationFieldImage,
                                     this->forwardJacobianMatrix);
      reg_resampleImage(this->currentFloating,
                        this->warped,
                        this->deformationFieldImage,
                        this->currentMask,
                        inter,
                        this->warpedPaddingValue,
                        this->measure_dti->GetActiveTimepoints(),
                        this->forwardJacobianMatrix);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::WarpFloatingImage");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_base<T>::Run()
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] %s::Run() called\n", this->executableName);
#endif

   if(!this->initialised) this->Initialise();

   // Compute the resolution of the progress bar
   float iProgressStep=1, nProgressSteps;
   nProgressSteps = this->levelToPerform*this->maxiterationNumber;

   // Loop over the different resolution level to perform
   for(this->currentLevel=0;
         this->currentLevel<this->levelToPerform;
         this->currentLevel++)
   {

      // Set the current input images
      if(this->usePyramid)
      {
         this->currentReference = this->referencePyramid[this->currentLevel];
         this->currentFloating = this->floatingPyramid[this->currentLevel];
         this->currentMask = this->maskPyramid[this->currentLevel];
      }
      else
      {
         this->currentReference = this->referencePyramid[0];
         this->currentFloating = this->floatingPyramid[0];
         this->currentMask = this->maskPyramid[0];
      }

      // Allocate image that depends on the reference image
      this->AllocateWarped();
      this->AllocateDeformationField();
      this->AllocateWarpedGradient();

      // The grid is refined if necessary
      T maxStepSize=this->InitialiseCurrentLevel();
      T currentSize = maxStepSize;
      T smallestSize = maxStepSize / (T)100.0;

      this->DisplayCurrentLevelParameters();

      // Allocate iamge that are required to compute the gradient
      this->AllocateVoxelBasedMeasureGradient();
      this->AllocateTransformationGradient();

      // Initialise the measures of similarity
      this->InitialiseSimilarity();

      // initialise the optimiser
      this->SetOptimiser();

      // Loop over the number of perturbation to do
      for(size_t perturbation=0;
            perturbation<=this->perturbationNumber;
            ++perturbation)
      {

         // Evalulate the objective function value
         this->UpdateBestObjFunctionValue();
         this->PrintInitialObjFunctionValue();

         // Iterate until convergence or until the max number of iteration is reach
         while(true)
         {

            if(currentSize==0)
               break;

            if(this->optimiser->GetCurrentIterationNumber()>=this->optimiser->GetMaxIterationNumber())
               break;

            // Compute the objective function gradient
            this->GetObjectiveFunctionGradient();

            // Normalise the gradient
            this->NormaliseGradient();

            // Initialise the line search initial step size
            currentSize=currentSize>maxStepSize?maxStepSize:currentSize;

            // A line search is performed
            this->optimiser->Optimise(maxStepSize,smallestSize,currentSize);

            // Update the obecjtive function variables and print some information
            this->PrintCurrentObjFunctionValue(currentSize);

            // Monitoring progression when f3d is ran as a library
            if(currentSize==0.f)
            {
               iProgressStep += this->optimiser->GetMaxIterationNumber() - 1 - this->optimiser->GetCurrentIterationNumber();
               if(funcProgressCallback && paramsProgressCallback)
               {
                  (*funcProgressCallback)(100.*iProgressStep/nProgressSteps,
                                          paramsProgressCallback);
               }
               break;
            }
            else
            {
               iProgressStep++;
               if(funcProgressCallback && paramsProgressCallback)
               {
                  (*funcProgressCallback)(100.*iProgressStep/nProgressSteps,
                                          paramsProgressCallback);
               }
            }
         } // while
         if(perturbation<this->perturbationNumber)
         {

            this->optimiser->Perturbation(smallestSize);
            currentSize=maxStepSize;
#ifdef NDEBUG
            if(this->verbose)
            {
#endif
               printf("[%s] Perturbation Step - The number of iteration is reset to 0\n",
                      this->executableName);
               printf("[%s] Perturbation Step - Every control point positions is altered by [-%g %g]\n",
                      this->executableName,
                      smallestSize,
                      smallestSize);

#ifdef NDEBUG
            }
#endif
         }
      } // perturbation loop

      // Final folding correction
      this->CorrectTransformation();

      // Some cleaning is performed
      delete this->optimiser;
      this->optimiser=NULL;
      this->ClearWarped();
      this->ClearDeformationField();
      this->ClearWarpedGradient();
      this->ClearVoxelBasedMeasureGradient();
      this->ClearTransformationGradient();
      if(this->usePyramid)
      {
         nifti_image_free(this->referencePyramid[this->currentLevel]);
         this->referencePyramid[this->currentLevel]=NULL;
         nifti_image_free(this->floatingPyramid[this->currentLevel]);
         this->floatingPyramid[this->currentLevel]=NULL;
         free(this->maskPyramid[this->currentLevel]);
         this->maskPyramid[this->currentLevel]=NULL;
      }
      else if(this->currentLevel==this->levelToPerform-1)
      {
         nifti_image_free(this->referencePyramid[0]);
         this->referencePyramid[0]=NULL;
         nifti_image_free(this->floatingPyramid[0]);
         this->floatingPyramid[0]=NULL;
         free(this->maskPyramid[0]);
         this->maskPyramid[0]=NULL;
      }
      this->ClearCurrentInputImage();

#ifdef NDEBUG
      if(this->verbose)
      {
#endif
         printf("[%s] Current registration level done\n", this->executableName);
         printf("[%s] --------------------------------------------------\n", this->executableName);
#ifdef NDEBUG
      }
#endif

   } // level this->levelToPerform

   if ( funcProgressCallback && paramsProgressCallback )
   {
      (*funcProgressCallback)( 100., paramsProgressCallback);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_base<T>::Run");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif // _REG_BASE_CPP
