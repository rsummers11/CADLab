/*
 *  _reg_f3_symd.cpp
 *
 *
 *  Created by Marc Modat on 10/11/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_SYM_CPP
#define _REG_F3D_SYM_CPP

#include "_reg_f3d_sym.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_sym<T>::reg_f3d_sym(int refTimePoint,int floTimePoint)
   :reg_f3d<T>::reg_f3d(refTimePoint,floTimePoint)
{
   this->executableName=(char *)"NiftyReg F3D SYM";

   this->backwardControlPointGrid=NULL;
   this->backwardWarped=NULL;
   this->backwardWarpedGradientImage=NULL;
   this->backwardDeformationFieldImage=NULL;
   this->backwardVoxelBasedMeasureGradientImage=NULL;
   this->backwardTransformationGradient=NULL;

   this->backwardProbaJointHistogram=NULL;
   this->backwardLogJointHistogram=NULL;

   this->floatingMaskImage=NULL;
   this->currentFloatingMask=NULL;
   this->floatingMaskPyramid=NULL;
   this->backwardActiveVoxelNumber=NULL;

   this->backwardJacobianMatrix=NULL;

   this->inverseConsistencyWeight=0.1;

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::reg_f3d_sym");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_sym<T>::~reg_f3d_sym()
{
   if(this->backwardControlPointGrid!=NULL)
   {
      nifti_image_free(this->backwardControlPointGrid);
      this->backwardControlPointGrid=NULL;
   }

   if(this->floatingMaskPyramid!=NULL)
   {
      if(this->usePyramid)
      {
         for(unsigned int i=0; i<this->levelToPerform; i++)
         {
            if(this->floatingMaskPyramid[i]!=NULL)
            {
               free(this->floatingMaskPyramid[i]);
               this->floatingMaskPyramid[i]=NULL;
            }
         }
      }
      else
      {
         if(this->floatingMaskPyramid[0]!=NULL)
         {
            free(this->floatingMaskPyramid[0]);
            this->floatingMaskPyramid[0]=NULL;
         }
      }
      free(this->floatingMaskPyramid);
      floatingMaskPyramid=NULL;
   }

   if(this->backwardActiveVoxelNumber!=NULL)
   {
      free(this->backwardActiveVoxelNumber);
      this->backwardActiveVoxelNumber=NULL;
   }

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::~reg_f3d_sym");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::SetFloatingMask(nifti_image *m)
{
   this->floatingMaskImage = m;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::~SetFloatingMask");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::SetInverseConsistencyWeight(T w)
{
   this->inverseConsistencyWeight = w;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::SetInverseConsistencyWeight");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d_sym<T>::InitialiseCurrentLevel()
{
   // Refine the control point grids if required
   if(this->currentLevel!=0)
   {
      reg_spline_refineControlPointGrid(this->controlPointGrid);
      reg_spline_refineControlPointGrid(this->backwardControlPointGrid);
   }

   // Set the mask images
   if(this->usePyramid)
   {
      this->currentMask = this->maskPyramid[this->currentLevel];
      this->currentFloatingMask = this->floatingMaskPyramid[this->currentLevel];
   }
   else
   {
      this->currentMask = this->maskPyramid[0];
      this->currentFloatingMask = this->floatingMaskPyramid[0];
   }
   // Define the initial step size for the gradient ascent optimisation
   T maxStepSize = this->currentReference->dx;
   maxStepSize = this->currentReference->dy>maxStepSize?this->currentReference->dy:maxStepSize;
   maxStepSize = this->currentFloating->dx>maxStepSize?this->currentFloating->dx:maxStepSize;
   maxStepSize = this->currentFloating->dy>maxStepSize?this->currentFloating->dy:maxStepSize;
   if(this->currentReference->ndim>2)
   {
      maxStepSize = (this->currentReference->dz>maxStepSize)?this->currentReference->dz:maxStepSize;
      maxStepSize = (this->currentFloating->dz>maxStepSize)?this->currentFloating->dz:maxStepSize;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::InitialiseCurrentLevel");
#endif
   return maxStepSize;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearCurrentInputImage()
{
   reg_f3d<T>::ClearCurrentInputImage();
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ClearCurrentInputImage");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateWarped()
{
   this->ClearWarped();

   reg_f3d<T>::AllocateWarped();
   if(this->currentFloating==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The floating image is not defined\n");
      reg_exit(1);
   }
   this->backwardWarped = nifti_copy_nim_info(this->currentFloating);
   this->backwardWarped->dim[0]=this->backwardWarped->ndim=this->currentReference->ndim;
   this->backwardWarped->dim[4]=this->backwardWarped->nt=this->currentReference->nt;
   this->backwardWarped->pixdim[4]=this->backwardWarped->dt=1.0;
   this->backwardWarped->nvox =
         (size_t)this->backwardWarped->nx *
         (size_t)this->backwardWarped->ny *
         (size_t)this->backwardWarped->nz *
         (size_t)this->backwardWarped->nt;
   this->backwardWarped->datatype = this->currentReference->datatype;
   this->backwardWarped->nbyper = this->currentReference->nbyper;
   this->backwardWarped->data = (void *)calloc(this->backwardWarped->nvox, this->backwardWarped->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::AllocateWarped");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearWarped()
{
   reg_f3d<T>::ClearWarped();
   if(this->backwardWarped!=NULL)
   {
      nifti_image_free(this->backwardWarped);
      this->backwardWarped=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ClearWarped");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateDeformationField()
{
   this->ClearDeformationField();

   reg_f3d<T>::AllocateDeformationField();
   if(this->currentFloating==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The floating image is not defined\n");
      reg_exit(1);
   }
   if(this->backwardControlPointGrid==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
      reg_exit(1);
   }
   this->backwardDeformationFieldImage = nifti_copy_nim_info(this->currentFloating);
   this->backwardDeformationFieldImage->dim[0]=this->backwardDeformationFieldImage->ndim=5;
   this->backwardDeformationFieldImage->dim[1]=this->backwardDeformationFieldImage->nx=this->currentFloating->nx;
   this->backwardDeformationFieldImage->dim[2]=this->backwardDeformationFieldImage->ny=this->currentFloating->ny;
   this->backwardDeformationFieldImage->dim[3]=this->backwardDeformationFieldImage->nz=this->currentFloating->nz;
   this->backwardDeformationFieldImage->dim[4]=this->backwardDeformationFieldImage->nt=1;
   this->backwardDeformationFieldImage->pixdim[4]=this->backwardDeformationFieldImage->dt=1.0;
   if(this->currentFloating->nz==1)
      this->backwardDeformationFieldImage->dim[5]=this->backwardDeformationFieldImage->nu=2;
   else this->backwardDeformationFieldImage->dim[5]=this->backwardDeformationFieldImage->nu=3;
   this->backwardDeformationFieldImage->pixdim[5]=this->backwardDeformationFieldImage->du=1.0;
   this->backwardDeformationFieldImage->dim[6]=this->backwardDeformationFieldImage->nv=1;
   this->backwardDeformationFieldImage->pixdim[6]=this->backwardDeformationFieldImage->dv=1.0;
   this->backwardDeformationFieldImage->dim[7]=this->backwardDeformationFieldImage->nw=1;
   this->backwardDeformationFieldImage->pixdim[7]=this->backwardDeformationFieldImage->dw=1.0;
   this->backwardDeformationFieldImage->nvox =
         (size_t)this->backwardDeformationFieldImage->nx *
         (size_t)this->backwardDeformationFieldImage->ny *
         (size_t)this->backwardDeformationFieldImage->nz *
         (size_t)this->backwardDeformationFieldImage->nt *
         (size_t)this->backwardDeformationFieldImage->nu;
   this->backwardDeformationFieldImage->nbyper = this->backwardControlPointGrid->nbyper;
   this->backwardDeformationFieldImage->datatype = this->backwardControlPointGrid->datatype;
   this->backwardDeformationFieldImage->data = (void *)calloc(this->backwardDeformationFieldImage->nvox,
                                                              this->backwardDeformationFieldImage->nbyper);
   this->backwardDeformationFieldImage->intent_code=NIFTI_INTENT_VECTOR;
   memset(this->backwardDeformationFieldImage->intent_name, 0, 16);
   strcpy(this->backwardDeformationFieldImage->intent_name,"NREG_TRANS");
   this->backwardDeformationFieldImage->intent_p1=DEF_FIELD;
   this->backwardDeformationFieldImage->scl_slope=1.f;
   this->backwardDeformationFieldImage->scl_inter=0.f;

   if(this->measure_dti!=NULL)
      this->backwardJacobianMatrix=(mat33 *)malloc(
            this->backwardDeformationFieldImage->nx *
            this->backwardDeformationFieldImage->ny *
            this->backwardDeformationFieldImage->nz *
            sizeof(mat33));

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::AllocateDeformationField");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearDeformationField()
{
   reg_f3d<T>::ClearDeformationField();
   if(this->backwardDeformationFieldImage!=NULL)
   {
      nifti_image_free(this->backwardDeformationFieldImage);
      this->backwardDeformationFieldImage=NULL;
   }
   if(this->backwardJacobianMatrix!=NULL)
   {
      free(this->backwardJacobianMatrix);
      this->backwardJacobianMatrix=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ClearDeformationField");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateWarpedGradient()
{
   this->ClearWarpedGradient();

   reg_f3d<T>::AllocateWarpedGradient();
   if(this->backwardDeformationFieldImage==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
      reg_exit(1);
   }
   this->backwardWarpedGradientImage = nifti_copy_nim_info(this->backwardDeformationFieldImage);
   this->backwardWarpedGradientImage->dim[0]=this->backwardWarpedGradientImage->ndim=5;
   this->backwardWarpedGradientImage->nt = this->backwardWarpedGradientImage->dim[4] = this->currentReference->nt;
   this->backwardWarpedGradientImage->nvox =
         (size_t)this->backwardWarpedGradientImage->nx *
         (size_t)this->backwardWarpedGradientImage->ny *
         (size_t)this->backwardWarpedGradientImage->nz *
         (size_t)this->backwardWarpedGradientImage->nt *
         (size_t)this->backwardWarpedGradientImage->nu;
   this->backwardWarpedGradientImage->data = (void *)calloc(this->backwardWarpedGradientImage->nvox,
                                                            this->backwardWarpedGradientImage->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::AllocateWarpedGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearWarpedGradient()
{
   reg_f3d<T>::ClearWarpedGradient();
   if(this->backwardWarpedGradientImage!=NULL)
   {
      nifti_image_free(this->backwardWarpedGradientImage);
      this->backwardWarpedGradientImage=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ClearWarpedGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateVoxelBasedMeasureGradient()
{
   this->ClearVoxelBasedMeasureGradient();

   reg_f3d<T>::AllocateVoxelBasedMeasureGradient();
   if(this->backwardDeformationFieldImage==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
      reg_exit(1);
   }
   this->backwardVoxelBasedMeasureGradientImage = nifti_copy_nim_info(this->backwardDeformationFieldImage);
   this->backwardVoxelBasedMeasureGradientImage->data =
         (void *)calloc(this->backwardVoxelBasedMeasureGradientImage->nvox,
                        this->backwardVoxelBasedMeasureGradientImage->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::AllocateVoxelBasedMeasureGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearVoxelBasedMeasureGradient()
{
   reg_f3d<T>::ClearVoxelBasedMeasureGradient();
   if(this->backwardVoxelBasedMeasureGradientImage!=NULL)
   {
      nifti_image_free(this->backwardVoxelBasedMeasureGradientImage);
      this->backwardVoxelBasedMeasureGradientImage=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ClearVoxelBasedMeasureGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateTransformationGradient()
{
   this->ClearTransformationGradient();

   reg_f3d<T>::AllocateTransformationGradient();
   if(this->backwardControlPointGrid==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
      reg_exit(1);
   }
   this->backwardTransformationGradient = nifti_copy_nim_info(this->backwardControlPointGrid);
   this->backwardTransformationGradient->data =
         (void *)calloc(this->backwardTransformationGradient->nvox,
                        this->backwardTransformationGradient->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::AllocateTransformationGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearTransformationGradient()
{
   reg_f3d<T>::ClearTransformationGradient();
   if(this->backwardTransformationGradient!=NULL)
      nifti_image_free(this->backwardTransformationGradient);
   this->backwardTransformationGradient=NULL;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ClearTransformationGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::CheckParameters()
{

   reg_f3d<T>::CheckParameters();

   // CHECK THE FLOATING MASK DIMENSION IF IT IS DEFINED
   if(this->floatingMaskImage!=NULL)
   {
      if(this->inputFloating->nx != this->floatingMaskImage->nx ||
            this->inputFloating->ny != this->floatingMaskImage->ny ||
            this->inputFloating->nz != this->floatingMaskImage->nz)
      {
         fprintf(stderr,"* The floating mask image has different x, y or z dimension than the floating image.\n");
         reg_exit(1);
      }
   }

   // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
   T penaltySum=
         this->bendingEnergyWeight
         +this->linearEnergyWeight0
         +this->linearEnergyWeight1
         +this->L2NormWeight
         +this->jacobianLogWeight
         +this->inverseConsistencyWeight;
   if(penaltySum>=1)
   {
      this->similarityWeight=0;
      this->bendingEnergyWeight /= penaltySum;
      this->linearEnergyWeight0 /= penaltySum;
      this->linearEnergyWeight1 /= penaltySum;
      this->L2NormWeight /= penaltySum;
      this->jacobianLogWeight /= penaltySum;
      this->inverseConsistencyWeight /= penaltySum;
   }
   else this->similarityWeight = 1.0 - penaltySum;

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::CheckParameters");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::Initialise()
{
   reg_f3d<T>::Initialise();

   if(this->inputControlPointGrid==NULL){
      // Define the spacing for the first level
      float gridSpacing[3] = {this->spacing[0],this->spacing[1],this->spacing[2]};
      if(this->spacing[0]<0)
         gridSpacing[0] *= -(this->inputReference->dx+this->inputFloating->dx)/2.f;
      if(this->spacing[1]<0)
         gridSpacing[1] *= -(this->inputReference->dy+this->inputFloating->dy)/2.f;
      if(this->spacing[2]<0)
         gridSpacing[2] *= -(this->inputReference->dz+this->inputFloating->dz)/2.f;
      gridSpacing[0] *= powf(2.0f, (float)(this->levelNumber-1));
      gridSpacing[1] *= powf(2.0f, (float)(this->levelNumber-1));
      gridSpacing[2] *= powf(2.0f, (float)(this->levelNumber-1));

      // Create the forward and backward control point grids
      reg_createSymmetricControlPointGrids<T>(&this->controlPointGrid,
                                              &this->backwardControlPointGrid,
                                              this->referencePyramid[0],
            this->floatingPyramid[0],
            this->affineTransformation,
            gridSpacing);
   }
   else{
      // The control point grid image is initialised with the provided grid
      this->controlPointGrid = nifti_copy_nim_info(this->inputControlPointGrid);
      this->controlPointGrid->data = (void *)malloc( this->controlPointGrid->nvox *
                                                     this->controlPointGrid->nbyper);
      if(this->inputControlPointGrid->num_ext>0)
         nifti_copy_extensions(this->controlPointGrid,this->inputControlPointGrid);
      memcpy( this->controlPointGrid->data, this->inputControlPointGrid->data,
              this->controlPointGrid->nvox * this->controlPointGrid->nbyper);
      // The final grid spacing is computed
      this->spacing[0] = this->controlPointGrid->dx / powf(2.0f, (float)(this->levelToPerform-1));
      this->spacing[1] = this->controlPointGrid->dy / powf(2.0f, (float)(this->levelToPerform-1));
      if(this->controlPointGrid->nz>1)
         this->spacing[2] = this->controlPointGrid->dz / powf(2.0f, (float)(this->levelToPerform-1));
      // The backward grid is derived from the forward
      this->backwardControlPointGrid=nifti_copy_nim_info(this->controlPointGrid);
      this->backwardControlPointGrid->data = (void *)malloc(this->backwardControlPointGrid->nvox *
                                                            this->backwardControlPointGrid->nbyper);
      if(this->controlPointGrid->num_ext>0)
         nifti_copy_extensions(this->backwardControlPointGrid,this->controlPointGrid);
      memcpy(this->backwardControlPointGrid->data,
             this->controlPointGrid->data,
             this->backwardControlPointGrid->nvox*this->backwardControlPointGrid->nbyper);
      reg_getDisplacementFromDeformation(this->backwardControlPointGrid);
      reg_tools_multiplyValueToImage(this->backwardControlPointGrid,this->backwardControlPointGrid,-1.f);
      reg_getDeformationFromDisplacement(this->backwardControlPointGrid);
      for(int i=0; i<this->backwardControlPointGrid->num_ext; ++i){
         mat44 tempMatrix=reg_mat44_inv(reinterpret_cast<mat44 *>(this->backwardControlPointGrid->ext_list[i].edata));
         memcpy(this->backwardControlPointGrid->ext_list[i].edata,
                &tempMatrix,
                sizeof(mat44));
      }
   }

   // Set the floating mask image pyramid
   if(this->usePyramid)
   {
      this->floatingMaskPyramid = (int **)malloc(this->levelToPerform*sizeof(int *));
      this->backwardActiveVoxelNumber= (int *)malloc(this->levelToPerform*sizeof(int));
   }
   else
   {
      this->floatingMaskPyramid = (int **)malloc(sizeof(int *));
      this->backwardActiveVoxelNumber= (int *)malloc(sizeof(int));
   }

   if(this->usePyramid)
   {
      if (this->floatingMaskImage!=NULL)
         reg_createMaskPyramid<T>(this->floatingMaskImage,
                                  this->floatingMaskPyramid,
                                  this->levelNumber,
                                  this->levelToPerform,
                                  this->backwardActiveVoxelNumber);
      else
      {
         for(unsigned int l=0; l<this->levelToPerform; ++l)
         {
            this->backwardActiveVoxelNumber[l]=this->floatingPyramid[l]->nx*this->floatingPyramid[l]->ny*this->floatingPyramid[l]->nz;
            this->floatingMaskPyramid[l]=(int *)calloc(backwardActiveVoxelNumber[l],sizeof(int));
         }
      }
   }
   else  // no pyramid
   {
      if (this->floatingMaskImage!=NULL)
         reg_createMaskPyramid<T>(this->floatingMaskImage, this->floatingMaskPyramid, 1, 1, this->backwardActiveVoxelNumber);
      else
      {
         this->backwardActiveVoxelNumber[0]=this->floatingPyramid[0]->nx*this->floatingPyramid[0]->ny*this->floatingPyramid[0]->nz;
         this->floatingMaskPyramid[0]=(int *)calloc(backwardActiveVoxelNumber[0],sizeof(int));
      }
   }

#ifdef NDEBUG
   if(this->verbose)
   {
#endif
      printf("[%s]\n", this->executableName);
      printf("[%s] Inverse consistency error penalty term weight: %g\n",
             this->executableName, this->inverseConsistencyWeight);
#ifdef NDEBUG
   }
#endif

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::Initialise");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetDeformationField()
{
   reg_spline_getDeformationField(this->controlPointGrid,
                                  this->deformationFieldImage,
                                  this->currentMask,
                                  false, //composition
                                  true // bspline
                                  );
   reg_spline_getDeformationField(this->backwardControlPointGrid,
                                  this->backwardDeformationFieldImage,
                                  this->currentFloatingMask,
                                  false, //composition
                                  true // bspline
                                  );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetDeformationField");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::WarpFloatingImage(int inter)
{
   // Compute the deformation fields
   this->GetDeformationField();

   // Resample the floating image
   if(this->measure_dti==NULL)
   {
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

   // Resample the reference image
   if(this->measure_dti==NULL)
   {
      reg_resampleImage(this->currentReference, // input image
                        this->backwardWarped, // warped input image
                        this->backwardDeformationFieldImage, // deformation field
                        this->currentFloatingMask, // mask
                        inter, // interpolation type
                        this->warpedPaddingValue); // padding value
   }
   else
   {
      reg_defField_getJacobianMatrix(this->backwardDeformationFieldImage,
                                     this->backwardJacobianMatrix);
      reg_resampleImage(this->currentReference, // input image
                        this->backwardWarped, // warped input image
                        this->backwardDeformationFieldImage, // deformation field
                        this->currentFloatingMask, // mask
                        inter, // interpolation type
                        this->warpedPaddingValue, // padding value
                        this->measure_dti->GetActiveTimepoints(),
                        this->backwardJacobianMatrix);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::WarpFloatingImage");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
   if (this->jacobianLogWeight<=0) return 0.;

   double forwardPenaltyTerm=reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(type);

   double backwardPenaltyTerm=0.;

   if(type==2)
   {
      backwardPenaltyTerm = reg_spline_getJacobianPenaltyTerm(this->backwardControlPointGrid,
                                                              this->currentFloating,
                                                              false);
   }
   else
   {
      backwardPenaltyTerm = reg_spline_getJacobianPenaltyTerm(this->backwardControlPointGrid,
                                                              this->currentFloating,
                                                              this->jacobianLogApproximation);
   }
   unsigned int maxit=5;
   if(type>0) maxit=20;
   unsigned int it=0;
   while(backwardPenaltyTerm!=backwardPenaltyTerm && it<maxit)
   {
      if(type==2)
      {
         backwardPenaltyTerm = reg_spline_correctFolding(this->backwardControlPointGrid,
                                                         this->currentFloating,
                                                         false);
      }
      else
      {
         backwardPenaltyTerm = reg_spline_correctFolding(this->backwardControlPointGrid,
                                                         this->currentFloating,
                                                         this->jacobianLogApproximation);
      }
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] Folding correction - Backward transformation\n");
#endif
      it++;
   }
   if(type>0 && it>0)
   {
      if(backwardPenaltyTerm!=backwardPenaltyTerm)
      {
         this->optimiser->RestoreBestDOF();
#ifndef NDEBUG
         fprintf(stderr, "[NiftyReg ERROR] The backward transformation folding correction scheme failed\n");
#endif
      }
      else
      {
#ifdef NDEBUG
         if(this->verbose)
         {
#endif
            printf("[%s] Backward transformation folding correction, %i step(s)\n", this->executableName, it);
#ifdef NDEBUG
         }
#endif
      }
   }
   backwardPenaltyTerm *= (double)this->jacobianLogWeight;

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ComputeJacobianBasedPenaltyTerm");
#endif
   return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeBendingEnergyPenaltyTerm()
{
   if (this->bendingEnergyWeight<=0) return 0.;

   double forwardPenaltyTerm=reg_f3d<T>::ComputeBendingEnergyPenaltyTerm();

   double value = reg_spline_approxBendingEnergy(this->backwardControlPointGrid);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ComputeBendingEnergyPenaltyTerm");
#endif
   return forwardPenaltyTerm + this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeLinearEnergyPenaltyTerm()
{
   if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0) return 0.;

   double forwardPenaltyTerm=reg_f3d<T>::ComputeLinearEnergyPenaltyTerm();

   double values_le[2]= {0.,0.};
   reg_spline_linearEnergy(this->backwardControlPointGrid, values_le);

   double backwardPenaltyTerm = this->linearEnergyWeight0*values_le[0] +
         this->linearEnergyWeight1*values_le[1];

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ComputeLinearEnergyPenaltyTerm");
#endif
   return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeL2NormDispPenaltyTerm()
{
   if(this->L2NormWeight<=0) return 0.;

   // Compute the L2 norm penalty term along the forward direction
   double forwardPenaltyTerm=reg_f3d<T>::ComputeL2NormDispPenaltyTerm();

   // Compute the L2 norm penalty term along the backward direction
   double backwardPenaltyTerm = (double)this->L2NormWeight *
         reg_spline_L2norm_displacement(this->backwardControlPointGrid);

   // Return the sum of the forward and backward squared L2 norm of the displacement
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::ComputeL2NormDispPenaltyTerm");
#endif
   return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetVoxelBasedGradient()
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
   //                             this->forwardJacobianMatrix,
   //                             this->warped);

   //        reg_getImageGradient(this->currentReference,
   //                             this->backwardWarpedGradientImage,
   //                             this->backwardDeformationFieldImage,
   //                             this->currentFloatingMask,
   //                             this->interpolation,
   //                             this->warpedPaddingValue,
   //                             this->measure_dti->GetActiveTimepoints(),
   //                             this->backwardJacobianMatrix,
   //                             this->backwardWarped);
   //    }
   //    else{
   reg_getImageGradient(this->currentFloating,
                        this->warpedGradientImage,
                        this->deformationFieldImage,
                        this->currentMask,
                        this->interpolation,
                        this->warpedPaddingValue);

   reg_getImageGradient(this->currentReference,
                        this->backwardWarpedGradientImage,
                        this->backwardDeformationFieldImage,
                        this->currentFloatingMask,
                        this->interpolation,
                        this->warpedPaddingValue);
   //    }
   // The voxel based gradient image is initialised with zeros
   reg_tools_multiplyValueToImage(this->voxelBasedMeasureGradientImage,
                                  this->voxelBasedMeasureGradientImage,
                                  0.f);
   reg_tools_multiplyValueToImage(this->backwardVoxelBasedMeasureGradientImage,
                                  this->backwardVoxelBasedMeasureGradientImage,
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
   reg_print_fct_debug("reg_f3d_sym<T>::GetVoxelBasedGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetSimilarityMeasureGradient()
{
   reg_f3d<T>::GetSimilarityMeasureGradient();

   // The voxel based sim measure gradient is convolved with a spline kernel
   // Convolution along the x axis
   float currentNodeSpacing[3];
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->backwardControlPointGrid->dx;
   bool activeAxis[3]= {1,0,0};
   reg_tools_kernelConvolution(this->backwardVoxelBasedMeasureGradientImage,
                               currentNodeSpacing,
                               1, // cubic spline kernel
                               NULL, // mask
                               NULL, // all volumes are active
                               activeAxis
                               );
   // Convolution along the y axis
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->backwardControlPointGrid->dy;
   activeAxis[0]=0;
   activeAxis[1]=1;
   reg_tools_kernelConvolution(this->backwardVoxelBasedMeasureGradientImage,
                               currentNodeSpacing,
                               1, // cubic spline kernel
                               NULL, // mask
                               NULL, // all volumes are active
                               activeAxis
                               );
   // Convolution along the z axis if required
   if(this->voxelBasedMeasureGradientImage->nz>1)
   {
      currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->backwardControlPointGrid->dz;
      activeAxis[1]=0;
      activeAxis[2]=1;
      reg_tools_kernelConvolution(this->backwardVoxelBasedMeasureGradientImage,
                                  currentNodeSpacing,
                                  1, // cubic spline kernel
                                  NULL, // mask
                                  NULL, // all volumes are active
                                  activeAxis
                                  );
   }
   // The backward node based sim measure gradient is extracted
   mat44 reorientation;
   if(this->currentReference->sform_code>0)
      reorientation = this->currentReference->sto_ijk;
   else reorientation = this->currentReference->qto_ijk;
   reg_voxelCentric2NodeCentric(this->backwardTransformationGradient,
                                this->backwardVoxelBasedMeasureGradientImage,
                                this->similarityWeight,
                                false, // no update
                                &reorientation // voxel to mm conversion
                                );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetSimilarityMeasureGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetJacobianBasedGradient()
{
   if(this->jacobianLogWeight<=0) return;

   reg_f3d<T>::GetJacobianBasedGradient();

   reg_spline_getJacobianPenaltyTermGradient(this->backwardControlPointGrid,
                                             this->currentFloating,
                                             this->backwardTransformationGradient,
                                             this->jacobianLogWeight,
                                             this->jacobianLogApproximation);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetJacobianBasedGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetBendingEnergyGradient()
{
   if(this->bendingEnergyWeight<=0) return;

   reg_f3d<T>::GetBendingEnergyGradient();
   reg_spline_approxBendingEnergyGradient(this->backwardControlPointGrid,
                                          this->backwardTransformationGradient,
                                          this->bendingEnergyWeight);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetBendingEnergyGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetLinearEnergyGradient()
{
   if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0) return;

   reg_f3d<T>::GetLinearEnergyGradient();

   reg_spline_linearEnergyGradient(this->backwardControlPointGrid,
                                   this->currentFloating,
                                   this->transformationGradient,
                                   this->linearEnergyWeight0,
                                   this->linearEnergyWeight1);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetLinearEnergyGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetL2NormDispGradient()
{
   if(this->L2NormWeight<=0) return;

   reg_f3d<T>::GetL2NormDispGradient();

   reg_spline_L2norm_dispGradient(this->backwardControlPointGrid,
                                  this->currentFloating,
                                  this->backwardTransformationGradient,
                                  this->L2NormWeight);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetL2NormDispGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::SetGradientImageToZero()
{
   reg_f3d<T>::SetGradientImageToZero();

   T* nodeGradPtr = static_cast<T *>(this->backwardTransformationGradient->data);
   for(size_t i=0; i<this->backwardTransformationGradient->nvox; ++i)
      *nodeGradPtr++=0;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::SetGradientImageToZero");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::SmoothGradient()
{
   if(this->gradientSmoothingSigma!=0)
   {
      reg_f3d<T>::SmoothGradient();
      // The gradient is smoothed using a Gaussian kernel if it is required
      float kernel = fabs(this->gradientSmoothingSigma);
      reg_tools_kernelConvolution(this->backwardTransformationGradient,
                                  &kernel,
                                  0);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::SmoothGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetApproximatedGradient()
{
   reg_f3d<T>::GetApproximatedGradient();

   // Loop over every control points
   T *gridPtr = static_cast<T *>(this->backwardControlPointGrid->data);
   T *gradPtr = static_cast<T *>(this->backwardTransformationGradient->data);
   T eps = this->currentFloating->dx/1000.f;
   for(size_t i=0; i<this->backwardControlPointGrid->nvox; i++)
   {
      T currentValue = this->optimiser->GetBestDOF_b()[i];
      gridPtr[i] = currentValue+eps;
      double valPlus = this->GetObjectiveFunctionValue();
      gridPtr[i] = currentValue-eps;
      double valMinus = this->GetObjectiveFunctionValue();
      gridPtr[i] = currentValue;
      gradPtr[i] = -(T)((valPlus - valMinus ) / (2.0*eps));
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetApproximatedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d_sym<T>::NormaliseGradient()
{
   // The forward gradient max length is computed
   T forwardMaxValue = reg_f3d<T>::NormaliseGradient();

   // The backward gradient max length is computed
   T maxGradValue=0;
   size_t voxNumber = this->backwardTransformationGradient->nx *
         this->backwardTransformationGradient->ny *
         this->backwardTransformationGradient->nz;
   T *bckPtrX = static_cast<T *>(this->backwardTransformationGradient->data);
   T *bckPtrY = &bckPtrX[voxNumber];
   if(this->backwardTransformationGradient->nz>1)
   {
      T *bckPtrZ = &bckPtrY[voxNumber];
      for(size_t i=0; i<voxNumber; i++)
      {
         T valX=0,valY=0,valZ=0;
         if(this->optimiseX==true)
            valX = *bckPtrX++;
         if(this->optimiseY==true)
            valY = *bckPtrY++;
         if(this->optimiseZ==true)
            valZ = *bckPtrZ++;
         T length = (T)(sqrt(valX*valX + valY*valY + valZ*valZ));
         maxGradValue = (length>maxGradValue)?length:maxGradValue;
      }
   }
   else
   {
      for(size_t i=0; i<voxNumber; i++)
      {
         T valX=0,valY=0;
         if(this->optimiseX==true)
            valX = *bckPtrX++;
         if(this->optimiseY==true)
            valY = *bckPtrY++;
         T length = (T)(sqrt(valX*valX + valY*valY));
         maxGradValue = (length>maxGradValue)?length:maxGradValue;
      }
   }

   // The largest value between the forward and backward gradient is kept
   maxGradValue = maxGradValue>forwardMaxValue?maxGradValue:forwardMaxValue;
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] Objective function gradient maximal length: %g\n", maxGradValue);
#endif

   // The forward gradient is normalised
   T *forPtrX = static_cast<T *>(this->transformationGradient->data);
   for(size_t i=0; i<this->transformationGradient->nvox; ++i)
   {
      *forPtrX++ /= maxGradValue;
   }
   // The backward gradient is normalised
   bckPtrX = static_cast<T *>(this->backwardTransformationGradient->data);
   for(size_t i=0; i<this->backwardTransformationGradient->nvox; ++i)
   {
      *bckPtrX++ /= maxGradValue;
   }

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::NormaliseGradient");
#endif
   // Returns the largest gradient distance
   return maxGradValue;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::GetObjectiveFunctionGradient()
{
   if(!this->useApproxGradient)
   {
      // Compute the gradient of the similarity measure
      if(this->similarityWeight>0)
      {
         this->WarpFloatingImage(this->interpolation);
         this->GetSimilarityMeasureGradient();
      }
      else
      {
         this->SetGradientImageToZero();
      }
   }
   else this->GetApproximatedGradient();
   this->optimiser->IncrementCurrentIterationNumber();

   // Smooth the gradient if require
   this->SmoothGradient();

   if(!this->useApproxGradient)
   {
      // Compute the penalty term gradients if required
      this->GetBendingEnergyGradient();
      this->GetJacobianBasedGradient();
      this->GetLinearEnergyGradient();
      this->GetL2NormDispGradient();
      this->GetInverseConsistencyGradient();
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetObjectiveFunctionGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::DisplayCurrentLevelParameters()
{
   reg_f3d<T>::DisplayCurrentLevelParameters();
#ifdef NDEBUG
   if(this->verbose)
   {
#endif
      printf("[%s] Current backward control point image\n", this->executableName);
      printf("[%s] \t* image dimension: %i x %i x %i\n", this->executableName,
             this->backwardControlPointGrid->nx, this->backwardControlPointGrid->ny,
             this->backwardControlPointGrid->nz);
      printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
             this->backwardControlPointGrid->dx, this->backwardControlPointGrid->dy,
             this->backwardControlPointGrid->dz);
#ifdef NDEBUG
   }
#endif

#ifndef NDEBUG

   if(this->backwardControlPointGrid->sform_code>0)
      reg_mat44_disp(&(this->backwardControlPointGrid->sto_xyz), (char *)"[NiftyReg DEBUG] Backward CPP sform");
   else reg_mat44_disp(&(this->backwardControlPointGrid->qto_xyz), (char *)"[NiftyReg DEBUG] Backward CPP qform");
#endif
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::DisplayCurrentLevelParameters");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::GetInverseConsistencyErrorField(bool forceAll)
{
   if (this->inverseConsistencyWeight<=0) return;

   // Compute both deformation fields
   if(this->similarityWeight<=0 || forceAll)
   {
      this->GetDeformationField();
   }
   // Compose the obtained deformation fields by the inverse transformations
   reg_spline_getDeformationField(this->backwardControlPointGrid,
                                  this->deformationFieldImage,
                                  this->currentMask,
                                  true, // composition
                                  true // use B-Spline
                                  );
   reg_spline_getDeformationField(this->controlPointGrid,
                                  this->backwardDeformationFieldImage,
                                  this->currentFloatingMask,
                                  true, // composition
                                  true // use B-Spline
                                  );
   // Convert the deformation fields into displacement
   reg_getDisplacementFromDeformation(this->deformationFieldImage);
   reg_getDisplacementFromDeformation(this->backwardDeformationFieldImage);

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetInverseConsistencyErrorField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
double reg_f3d_sym<T>::GetInverseConsistencyPenaltyTerm()
{
   if (this->inverseConsistencyWeight<=0) return 0.;

   this->GetInverseConsistencyErrorField(false);

   double ferror=0.;
   size_t voxelNumber=this->deformationFieldImage->nx *
         this->deformationFieldImage->ny *
         this->deformationFieldImage->nz;
   T *dispPtrX=static_cast<T *>(this->deformationFieldImage->data);
   T *dispPtrY=&dispPtrX[voxelNumber];
   if(this->deformationFieldImage->nz>1)
   {
      T *dispPtrZ=&dispPtrY[voxelNumber];
      for(size_t i=0; i<voxelNumber; ++i)
      {
         if(this->currentMask[i]>-1)
         {
            double dist=reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]) + reg_pow2(dispPtrZ[i]);
            ferror += dist;
         }
      }
   }
   else
   {
      for(size_t i=0; i<voxelNumber; ++i)
      {
         if(this->currentMask[i]>-1)
         {
            double dist=reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]);
            ferror += dist;
         }
      }
   }

   double berror=0.;
   voxelNumber=this->backwardDeformationFieldImage->nx *
         this->backwardDeformationFieldImage->ny *
         this->backwardDeformationFieldImage->nz;
   dispPtrX=static_cast<T *>(this->backwardDeformationFieldImage->data);
   dispPtrY=&dispPtrX[voxelNumber];
   if(this->backwardDeformationFieldImage->nz>1)
   {
      T *dispPtrZ=&dispPtrY[voxelNumber];
      for(size_t i=0; i<voxelNumber; ++i)
      {
         if(this->currentFloatingMask[i]>-1)
         {
            double dist=reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]) + reg_pow2(dispPtrZ[i]);
            berror += dist;
         }
      }
   }
   else
   {
      for(size_t i=0; i<voxelNumber; ++i)
      {
         if(this->currentFloatingMask[i]>-1)
         {
            double dist=reg_pow2(dispPtrX[i]) + reg_pow2(dispPtrY[i]);
            berror += dist;
         }
      }
   }
   double error = ferror/double(this->activeVoxelNumber[this->currentLevel])
         + berror / (double)(this->backwardActiveVoxelNumber[this->currentLevel]);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetInverseConsistencyPenaltyTerm");
#endif
   return double(this->inverseConsistencyWeight) * error;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::GetInverseConsistencyGradient()
{
   if(this->inverseConsistencyWeight<=0) return;

   // Note: I simplified the gradient computation in order to include
   // only d(B(F(x)))/d(forwardNode) and d(F(B(x)))/d(backwardNode)
   // I ignored d(F(B(x)))/d(forwardNode) and d(B(F(x)))/d(backwardNode)
   // cause it would only be an approximation since I don't have the
   // real inverses
   this->GetInverseConsistencyErrorField(true);

   // The forward inverse consistency field is masked
   size_t forwardVoxelNumber=
         this->deformationFieldImage->nx *
         this->deformationFieldImage->ny *
         this->deformationFieldImage->nz ;
   T *defPtrX=static_cast<T* >(this->deformationFieldImage->data);
   T *defPtrY=&defPtrX[forwardVoxelNumber];
   T *defPtrZ=&defPtrY[forwardVoxelNumber];
   for(size_t i=0; i<forwardVoxelNumber; ++i)
   {
      if(this->currentMask[i]<0)
      {
         defPtrX[i]=0;
         defPtrY[i]=0;
         if(this->deformationFieldImage->nz>1)
            defPtrZ[i]=0;
      }
   }
   // The backward inverse consistency field is masked
   size_t backwardVoxelNumber =
         this->backwardDeformationFieldImage->nx *
         this->backwardDeformationFieldImage->ny *
         this->backwardDeformationFieldImage->nz ;
   defPtrX=static_cast<T* >(this->backwardDeformationFieldImage->data);
   defPtrY=&defPtrX[backwardVoxelNumber];
   defPtrZ=&defPtrY[backwardVoxelNumber];
   for(size_t i=0; i<backwardVoxelNumber; ++i)
   {
      if(this->currentFloatingMask[i]<0)
      {
         defPtrX[i]=0;
         defPtrY[i]=0;
         if(this->backwardDeformationFieldImage->nz>1)
            defPtrZ[i]=0;
      }
   }

   // We convolve the inverse consistency map with a cubic B-Spline kernel
   // Convolution along the x axis
   float currentNodeSpacing[3];
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dx;
   bool activeAxis[3]= {1,0,0};
   reg_tools_kernelConvolution(this->deformationFieldImage,
                               currentNodeSpacing,
                               1, // cubic spline kernel
                               NULL, // all volumes are active
                               activeAxis
                               );
   // Convolution along the y axis
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dy;
   activeAxis[0]=0;
   activeAxis[1]=1;
   reg_tools_kernelConvolution(this->deformationFieldImage,
                               currentNodeSpacing,
                               1, // cubic spline kernel
                               NULL, // all volumes are active
                               activeAxis
                               );
   // Convolution along the z axis if required
   if(this->voxelBasedMeasureGradientImage->nz>1)
   {
      currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dz;
      activeAxis[1]=0;
      activeAxis[2]=1;
      reg_tools_kernelConvolution(this->deformationFieldImage,
                                  currentNodeSpacing,
                                  1, // cubic spline kernel
                                  NULL, // all volumes are active
                                  activeAxis
                                  );
   }
   // The forward inverse consistency gradient is extracted at the node position
   reg_voxelCentric2NodeCentric(this->transformationGradient,
                                this->deformationFieldImage,
                                2.f * this->inverseConsistencyWeight,
                                true, // update the current value
                                NULL // no voxel to mm conversion
                                );

   // We convolve the inverse consistency map with a cubic B-Spline kernel
   // Convolution along the x axis
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->backwardControlPointGrid->dx;
   activeAxis[0]=1;
   activeAxis[1]=0;
   activeAxis[2]=0;
   reg_tools_kernelConvolution(this->backwardDeformationFieldImage,
                               currentNodeSpacing,
                               1, // cubic spline kernel
                               NULL, // all volumes are active
                               activeAxis
                               );
   // Convolution along the y axis
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->backwardControlPointGrid->dy;
   activeAxis[0]=0;
   activeAxis[1]=1;
   reg_tools_kernelConvolution(this->backwardDeformationFieldImage,
                               currentNodeSpacing,
                               1, // cubic spline kernel
                               NULL, // all volumes are active
                               activeAxis
                               );
   // Convolution along the z axis if required
   if(this->voxelBasedMeasureGradientImage->nz>1)
   {
      currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->backwardControlPointGrid->dz;
      activeAxis[1]=0;
      activeAxis[2]=1;
      reg_tools_kernelConvolution(this->backwardDeformationFieldImage,
                                  currentNodeSpacing,
                                  1, // cubic spline kernel
                                  NULL, // all volumes are active
                                  activeAxis
                                  );
   }
   // The backward inverse consistency gradient is extracted at the node position
   reg_voxelCentric2NodeCentric(this->backwardTransformationGradient,
                                this->backwardDeformationFieldImage,
                                2.f * this->inverseConsistencyWeight,
                                true, // update the current value
                                NULL // no voxel to mm conversion
                                );

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetInverseConsistencyGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::UpdateParameters(float scale)
{
   // Update first the forward transformation
   reg_f3d<T>::UpdateParameters(scale);

   // Create some pointers to the relevant arrays
   T *currentDOF_b=this->optimiser->GetCurrentDOF_b();
   T *bestDOF_b=this->optimiser->GetBestDOF_b();
   T *gradient_b=this->optimiser->GetGradient_b();

   // Update the control point position
   if(this->optimiser->GetOptimiseX()==true &&
         this->optimiser->GetOptimiseY()==true &&
         this->optimiser->GetOptimiseZ()==true)
   {
      // Update the values for all axis displacement
      for(size_t i=0; i<this->optimiser->GetDOFNumber_b(); ++i)
      {
         currentDOF_b[i] = bestDOF_b[i] + scale * gradient_b[i];
      }
   }
   else
   {
      size_t voxNumber_b = this->optimiser->GetVoxNumber_b();
      // Update the values for the x-axis displacement
      if(this->optimiser->GetOptimiseX()==true)
      {
         for(size_t i=0; i<voxNumber_b; ++i)
         {
            currentDOF_b[i] =bestDOF_b[i] + scale * gradient_b[i];
         }
      }
      // Update the values for the y-axis displacement
      if(this->optimiser->GetOptimiseY()==true)
      {
         T *currentDOFY_b=&currentDOF_b[voxNumber_b];
         T *bestDOFY_b=&bestDOF_b[voxNumber_b];
         T *gradientY_b=&gradient_b[voxNumber_b];
         for(size_t i=0; i<voxNumber_b; ++i)
         {
            currentDOFY_b[i] = bestDOFY_b[i] + scale * gradientY_b[i];
         }
      }
      // Update the values for the z-axis displacement
      if(this->optimiser->GetOptimiseZ()==true && this->optimiser->GetNDim()>2)
      {
         T *currentDOFZ_b=&currentDOF_b[2*voxNumber_b];
         T *bestDOFZ_b=&bestDOF_b[2*voxNumber_b];
         T *gradientZ_b=&gradient_b[2*voxNumber_b];
         for(size_t i=0; i<voxNumber_b; ++i)
         {
            currentDOFZ_b[i] = bestDOFZ_b[i] + scale * gradientZ_b[i];
         }
      }
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::UpdateParameters");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::SetOptimiser()
{
   if(this->useConjGradient)
      this->optimiser=new reg_conjugateGradient<T>();
   else this->optimiser=new reg_optimiser<T>();
   this->optimiser->Initialise(this->controlPointGrid->nvox,
                               this->controlPointGrid->nz>1?3:2,
                               this->optimiseX,
                               this->optimiseY,
                               this->optimiseZ,
                               this->maxiterationNumber,
                               0, // currentIterationNumber
                               this,
                               static_cast<T *>(this->controlPointGrid->data),
                               static_cast<T *>(this->transformationGradient->data),
                               this->backwardControlPointGrid->nvox,
                               static_cast<T *>(this->backwardControlPointGrid->data),
                               static_cast<T *>(this->backwardTransformationGradient->data));
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::SetOptimiser");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::PrintCurrentObjFunctionValue(T currentSize)
{
   if(!this->verbose) return;

   printf("[%s] [%i] Current objective function: %g",
          this->executableName,
          (int)this->optimiser->GetCurrentIterationNumber(),
          this->optimiser->GetBestObjFunctionValue());
   printf(" = (wSIM)%g", this->bestWMeasure);
   if(this->bendingEnergyWeight>0)
      printf(" - (wBE)%.2e", this->bestWBE);
   if(this->linearEnergyWeight0>0 || this->linearEnergyWeight1>0)
      printf(" - (wLE)%.2e", this->bestWLE);
   if(this->L2NormWeight>0)
      printf(" - (wL2)%.2e", this->bestWL2);
   if(this->jacobianLogWeight>0)
      printf(" - (wJAC)%.2e", this->bestWJac);
   if(this->inverseConsistencyWeight>0)
      printf(" - (wIC)%.2e", this->bestIC);
   printf(" [+ %g mm]\n", currentSize);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::PrintCurrentObjFunctionValue");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::UpdateBestObjFunctionValue()
{
   reg_f3d<T>::UpdateBestObjFunctionValue();
   this->bestIC=this->currentIC;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::UpdateBestObjFunctionValue");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::PrintInitialObjFunctionValue()
{
   if(!this->verbose) return;
   reg_f3d<T>::PrintInitialObjFunctionValue();
   printf("[%s] Initial Inverse consistency value: %g\n", this->executableName, this->bestIC);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::PrintInitialObjFunctionValue");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::GetObjectiveFunctionValue()
{
   this->currentWJac = this->ComputeJacobianBasedPenaltyTerm(1); // 20 iterations

   this->currentWBE = this->ComputeBendingEnergyPenaltyTerm();

   this->currentWLE = this->ComputeLinearEnergyPenaltyTerm();

   this->currentWL2 = this->ComputeL2NormDispPenaltyTerm();

   // Compute initial similarity measure
   this->currentWMeasure = 0.0;
   if(this->similarityWeight>0)
   {
      this->WarpFloatingImage(this->interpolation);
      this->currentWMeasure = this->ComputeSimilarityMeasure();
   }

   // Compute the Inverse consistency penalty term if required
   this->currentIC = this->GetInverseConsistencyPenaltyTerm();

#ifndef NDEBUG
   printf("[NiftyReg DEBUG] (wMeasure) %g | (wBE) %g | (wLE) %g | (wL2) %g | (wJac) %g | (wIC) %g \n",
          this->currentWMeasure,
          this->currentWBE,
          this->currentWLE,
          this->currentWL2,
          this->currentWJac,
          this->currentIC);
#endif

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetObjectiveFunctionValue");
#endif
   // Store the global objective function value
   return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWL2 - this->currentWJac - this->currentIC;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::InitialiseSimilarity()
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
                                           this->voxelBasedMeasureGradientImage,
                                           this->currentFloatingMask,
                                           this->backwardWarped,
                                           this->backwardWarpedGradientImage,
                                           this->backwardVoxelBasedMeasureGradientImage
                                           );

   if(this->measure_multichannel_nmi!=NULL)
      this->measure_multichannel_nmi->InitialiseMeasure(this->currentReference,
                                                        this->currentFloating,
                                                        this->currentMask,
                                                        this->warped,
                                                        this->warpedGradientImage,
                                                        this->voxelBasedMeasureGradientImage,
                                                        this->currentFloatingMask,
                                                        this->backwardWarped,
                                                        this->backwardWarpedGradientImage,
                                                        this->backwardVoxelBasedMeasureGradientImage
                                                        );

   if(this->measure_ssd!=NULL)
      this->measure_ssd->InitialiseMeasure(this->currentReference,
                                           this->currentFloating,
                                           this->currentMask,
                                           this->warped,
                                           this->warpedGradientImage,
                                           this->voxelBasedMeasureGradientImage,
                                           this->currentFloatingMask,
                                           this->backwardWarped,
                                           this->backwardWarpedGradientImage,
                                           this->backwardVoxelBasedMeasureGradientImage
                                           );

   if(this->measure_kld!=NULL)
      this->measure_kld->InitialiseMeasure(this->currentReference,
                                           this->currentFloating,
                                           this->currentMask,
                                           this->warped,
                                           this->warpedGradientImage,
                                           this->voxelBasedMeasureGradientImage,
                                           this->currentFloatingMask,
                                           this->backwardWarped,
                                           this->backwardWarpedGradientImage,
                                           this->backwardVoxelBasedMeasureGradientImage
                                           );

   if(this->measure_lncc!=NULL)
      this->measure_lncc->InitialiseMeasure(this->currentReference,
                                            this->currentFloating,
                                            this->currentMask,
                                            this->warped,
                                            this->warpedGradientImage,
                                            this->voxelBasedMeasureGradientImage,
                                            this->currentFloatingMask,
                                            this->backwardWarped,
                                            this->backwardWarpedGradientImage,
                                            this->backwardVoxelBasedMeasureGradientImage
                                            );

   if(this->measure_dti!=NULL)
      this->measure_dti->InitialiseMeasure(this->currentReference,
                                           this->currentFloating,
                                           this->currentMask,
                                           this->warped,
                                           this->warpedGradientImage,
                                           this->voxelBasedMeasureGradientImage,
                                           this->currentFloatingMask,
                                           this->backwardWarped,
                                           this->backwardWarpedGradientImage,
                                           this->backwardVoxelBasedMeasureGradientImage
                                           );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::InitialiseSimilarity");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image **reg_f3d_sym<T>::GetWarpedImage()
{
   // The initial images are used
   if(this->inputReference==NULL ||
         this->inputFloating==NULL ||
         this->controlPointGrid==NULL ||
         this->backwardControlPointGrid==NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] reg_f3d_sym::GetWarpedImage()\n");
      fprintf(stderr," * The reference, floating and both control point grid images have to be defined\n");
   }

   reg_f3d_sym<T>::currentReference = this->inputReference;
   reg_f3d_sym<T>::currentFloating = this->inputFloating;
   reg_f3d_sym<T>::currentMask = NULL;
   reg_f3d_sym<T>::currentFloatingMask = NULL;

   reg_f3d_sym<T>::AllocateWarped();
   reg_f3d_sym<T>::AllocateDeformationField();

   reg_f3d_sym<T>::WarpFloatingImage(3); // cubic spline interpolation

   reg_f3d_sym<T>::ClearDeformationField();

   nifti_image **resultImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
   resultImage[0] = nifti_copy_nim_info(this->warped);
   resultImage[0]->cal_min=this->inputFloating->cal_min;
   resultImage[0]->cal_max=this->inputFloating->cal_max;
   resultImage[0]->scl_slope=this->inputFloating->scl_slope;
   resultImage[0]->scl_inter=this->inputFloating->scl_inter;
   resultImage[0]->data=(void *)malloc(resultImage[0]->nvox*resultImage[0]->nbyper);
   memcpy(resultImage[0]->data, this->warped->data, resultImage[0]->nvox*resultImage[0]->nbyper);

   resultImage[1] = nifti_copy_nim_info(this->backwardWarped);
   resultImage[1]->cal_min=this->inputReference->cal_min;
   resultImage[1]->cal_max=this->inputReference->cal_max;
   resultImage[1]->scl_slope=this->inputReference->scl_slope;
   resultImage[1]->scl_inter=this->inputReference->scl_inter;
   resultImage[1]->data=(void *)malloc(resultImage[1]->nvox*resultImage[1]->nbyper);
   memcpy(resultImage[1]->data, this->backwardWarped->data, resultImage[1]->nvox*resultImage[1]->nbyper);

   reg_f3d_sym<T>::ClearWarped();
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetWarpedImage");
#endif
   return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image * reg_f3d_sym<T>::GetBackwardControlPointPositionImage()
{
   // Create a control point grid nifti image
   nifti_image *returnedControlPointGrid = nifti_copy_nim_info(this->backwardControlPointGrid);
   // Allocate the new image data array
   returnedControlPointGrid->data=(void *)malloc(returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
   // Copy the final backward control point grid image
   memcpy(returnedControlPointGrid->data, this->backwardControlPointGrid->data,
          returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
   // Return the new control point grid
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d_sym<T>::GetBackwardControlPointPositionImage");
#endif
   return returnedControlPointGrid;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
