/**
 *  _reg_f3d.cpp
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_CPP
#define _REG_F3D_CPP

#include "_reg_f3d.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d<T>::reg_f3d(int refTimePoint,int floTimePoint)
   : reg_base<T>::reg_base(refTimePoint,floTimePoint)
{

   this->executableName=(char *)"NiftyReg F3D";
   this->inputControlPointGrid=NULL; // pointer to external
   this->controlPointGrid=NULL;
   this->bendingEnergyWeight=0.005;
   this->linearEnergyWeight0=0.;
   this->linearEnergyWeight1=0.;
   this->L2NormWeight=0.;
   this->jacobianLogWeight=0.;
   this->jacobianLogApproximation=true;
   this->spacing[0]=-5;
   this->spacing[1]=std::numeric_limits<T>::quiet_NaN();
   this->spacing[2]=std::numeric_limits<T>::quiet_NaN();
   this->useConjGradient=true;
   this->useApproxGradient=false;

//    this->approxParzenWindow=true;

   this->transformationGradient=NULL;

   this->gridRefinement=true;

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::reg_f3d");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d<T>::~reg_f3d()
{
   this->ClearTransformationGradient();
   if(this->controlPointGrid!=NULL)
   {
      nifti_image_free(this->controlPointGrid);
      this->controlPointGrid=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::~reg_f3d");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetControlPointGridImage(nifti_image *cp)
{
   this->inputControlPointGrid = cp;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetControlPointGridImage");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetBendingEnergyWeight(T be)
{
   this->bendingEnergyWeight = be;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetBendingEnergyWeight");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetLinearEnergyWeights(T w0, T w1)
{
   this->linearEnergyWeight0=w0;
   this->linearEnergyWeight1=w1;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetLinearEnergyWeights");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetL2NormDisplacementWeight(T w)
{
   this->L2NormWeight=w;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetL2NormDisplacementWeight");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetJacobianLogWeight(T j)
{
   this->jacobianLogWeight = j;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetJacobianLogWeight");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::ApproximateJacobianLog()
{
   this->jacobianLogApproximation = true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ApproximateJacobianLog");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::DoNotApproximateJacobianLog()
{
   this->jacobianLogApproximation = false;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::DoNotApproximateJacobianLog");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::SetSpacing(unsigned int i, T s)
{
   this->spacing[i] = s;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetSpacing");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d<T>::InitialiseCurrentLevel()
{
   // Set the initial step size for the gradient ascent
   T maxStepSize = this->currentReference->dx>this->currentReference->dy?this->currentReference->dx:this->currentReference->dy;
   if(this->currentReference->ndim>2)
      maxStepSize = (this->currentReference->dz>maxStepSize)?this->currentReference->dz:maxStepSize;

   // Refine the control point grid if required
   if(this->gridRefinement==true)
   {
      if(this->currentLevel==0)
         this->bendingEnergyWeight = this->bendingEnergyWeight / static_cast<T>(powf(16.0f, this->levelToPerform-1));
      else
      {
         reg_spline_refineControlPointGrid(this->controlPointGrid,this->currentReference);
         this->bendingEnergyWeight = this->bendingEnergyWeight * static_cast<T>(16);
      }
   }

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::InitialiseCurrentLevel");
#endif
   return maxStepSize;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::AllocateTransformationGradient()
{
   if(this->controlPointGrid==NULL)
   {
      fprintf(stderr, "[NiftyReg ERROR] The control point image is not defined\n");
      reg_exit(1);
   }
   reg_f3d<T>::ClearTransformationGradient();
   this->transformationGradient = nifti_copy_nim_info(this->controlPointGrid);
   this->transformationGradient->data = (void *)calloc(this->transformationGradient->nvox,
                                        this->transformationGradient->nbyper);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::AllocateTransformationGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::ClearTransformationGradient()
{
   if(this->transformationGradient!=NULL)
   {
      nifti_image_free(this->transformationGradient);
      this->transformationGradient=NULL;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ClearTransformationGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::CheckParameters()
{
   reg_base<T>::CheckParameters();

   // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
   if(strcmp(this->executableName,"NiftyReg F3D")==0 ||
         strcmp(this->executableName,"NiftyReg F3D GPU")==0)
   {
      T penaltySum=this->bendingEnergyWeight +
                   this->linearEnergyWeight0 +
                   this->linearEnergyWeight1 +
                   this->L2NormWeight +
                   this->jacobianLogWeight;
      if(penaltySum>=1.0)
      {
         this->similarityWeight=0;
         this->similarityWeight /= penaltySum;
         this->bendingEnergyWeight /= penaltySum;
         this->linearEnergyWeight0 /= penaltySum;
         this->linearEnergyWeight1 /= penaltySum;
         this->L2NormWeight /= penaltySum;
         this->jacobianLogWeight /= penaltySum;
      }
      else this->similarityWeight=1.0 - penaltySum;
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::CheckParameters");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::Initialise()
{
   if(this->initialised) return;

   reg_base<T>::Initialise();

   // DETERMINE THE GRID SPACING AND CREATE THE GRID
   if(this->inputControlPointGrid==NULL)
   {

      // Set the spacing along y and z if undefined. Their values are set to match
      // the spacing along the x axis
      if(this->spacing[1]!=this->spacing[1]) this->spacing[1]=this->spacing[0];
      if(this->spacing[2]!=this->spacing[2]) this->spacing[2]=this->spacing[0];

      /* Convert the spacing from voxel to mm if necessary */
      float spacingInMillimeter[3]= {this->spacing[0],this->spacing[1],this->spacing[2]};
      if(this->usePyramid)
      {
         if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dx;
         if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dy;
         if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->referencePyramid[this->levelToPerform-1]->dz;
      }
      else
      {
         if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->referencePyramid[0]->dx;
         if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->referencePyramid[0]->dy;
         if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->referencePyramid[0]->dz;
      }

      // Define the spacing for the first level
      float gridSpacing[3];
      gridSpacing[0] = spacingInMillimeter[0] * powf(2.0f, (float)(this->levelToPerform-1));
      gridSpacing[1] = spacingInMillimeter[1] * powf(2.0f, (float)(this->levelToPerform-1));
      gridSpacing[2] = 1.0f;
      if(this->referencePyramid[0]->nz>1)
         gridSpacing[2] = spacingInMillimeter[2] * powf(2.0f, (float)(this->levelToPerform-1));

      // Create and allocate the control point image
      reg_createControlPointGrid<T>(&this->controlPointGrid,
                                    this->referencePyramid[0],
                                    gridSpacing);

      // The control point position image is initialised with the affine transformation
      if(this->affineTransformation==NULL)
      {
         memset(this->controlPointGrid->data,0,
                this->controlPointGrid->nvox*this->controlPointGrid->nbyper);
         reg_tools_multiplyValueToImage(this->controlPointGrid,this->controlPointGrid,0.f);
         reg_getDeformationFromDisplacement(this->controlPointGrid);
      }
      else reg_affine_getDeformationField(this->affineTransformation, this->controlPointGrid);
   }
   else
   {
      // The control point grid image is initialised with the provided grid
      this->controlPointGrid = nifti_copy_nim_info(this->inputControlPointGrid);
      this->controlPointGrid->data = (void *)malloc( this->controlPointGrid->nvox *
                                     this->controlPointGrid->nbyper);
      memcpy( this->controlPointGrid->data, this->inputControlPointGrid->data,
              this->controlPointGrid->nvox * this->controlPointGrid->nbyper);
      // The final grid spacing is computed
      this->spacing[0] = this->controlPointGrid->dx / powf(2.0f, (float)(this->levelToPerform-1));
      this->spacing[1] = this->controlPointGrid->dy / powf(2.0f, (float)(this->levelToPerform-1));
      if(this->controlPointGrid->nz>1)
         this->spacing[2] = this->controlPointGrid->dz / powf(2.0f, (float)(this->levelToPerform-1));
   }
#ifdef NDEBUG
   if(this->verbose)
   {
#endif
      // Print out some global information about the registration
      printf("[%s] **************************************************\n", this->executableName);
      printf("[%s] INPUT PARAMETERS\n", this->executableName);
      printf("[%s] **************************************************\n", this->executableName);
      printf("[%s] Reference image:\n", this->executableName);
      printf("[%s] \t* name: %s\n", this->executableName, this->inputReference->fname);
      printf("[%s] \t* image dimension: %i x %i x %i x %i\n", this->executableName,
             this->inputReference->nx, this->inputReference->ny,
             this->inputReference->nz, this->inputReference->nt);
      printf("[%s] \t* image spacing: %g x %g x %g mm\n",
             this->executableName, this->inputReference->dx,
             this->inputReference->dy, this->inputReference->dz);
      for(int i=0; i<this->inputReference->nt; i++)
      {
         printf("[%s] \t* intensity threshold for timepoint %i/%i: [%.2g %.2g]\n", this->executableName,
                i+1, this->inputReference->nt, this->referenceThresholdLow[i],this->referenceThresholdUp[i]);
         if(this->measure_nmi!=NULL)
            if(this->measure_nmi->GetActiveTimepoints()[i])
               printf("[%s] \t* binnining size for timepoint %i/%i: %i\n", this->executableName,
                      i+1, this->inputFloating->nt, this->measure_nmi->GetReferenceBinNumber()[i]-4);
      }
      printf("[%s] \t* gaussian smoothing sigma: %g\n", this->executableName, this->referenceSmoothingSigma);
      printf("[%s]\n", this->executableName);
      printf("[%s] Floating image:\n", this->executableName);
      printf("[%s] \t* name: %s\n", this->executableName, this->inputFloating->fname);
      printf("[%s] \t* image dimension: %i x %i x %i x %i\n", this->executableName,
             this->inputFloating->nx, this->inputFloating->ny,
             this->inputFloating->nz, this->inputFloating->nt);
      printf("[%s] \t* image spacing: %g x %g x %g mm\n",
             this->executableName, this->inputFloating->dx,
             this->inputFloating->dy, this->inputFloating->dz);
      for(int i=0; i<this->inputFloating->nt; i++)
      {
         printf("[%s] \t* intensity threshold for timepoint %i/%i: [%.2g %.2g]\n", this->executableName,
                i+1, this->inputFloating->nt, this->floatingThresholdLow[i],this->floatingThresholdUp[i]);
         if(this->measure_nmi!=NULL)
            if(this->measure_nmi->GetActiveTimepoints()[i])
               printf("[%s] \t* binnining size for timepoint %i/%i: %i\n", this->executableName,
                      i+1, this->inputFloating->nt, this->measure_nmi->GetFloatingBinNumber()[i]-4);
      }
      printf("[%s] \t* gaussian smoothing sigma: %g\n",
             this->executableName, this->floatingSmoothingSigma);
      printf("[%s]\n", this->executableName);
      printf("[%s] Warped image padding value: %g\n", this->executableName, this->warpedPaddingValue);
      printf("[%s]\n", this->executableName);
      printf("[%s] Level number: %i\n", this->executableName, this->levelNumber);
      if(this->levelNumber!=this->levelToPerform)
         printf("[%s] \t* Level to perform: %i\n", this->executableName, this->levelToPerform);
      printf("[%s]\n", this->executableName);
      printf("[%s] Maximum iteration number per level: %i\n", this->executableName, (int)this->maxiterationNumber);
      printf("[%s]\n", this->executableName);
      printf("[%s] Final spacing in mm: %g %g %g\n", this->executableName,
             this->spacing[0], this->spacing[1], this->spacing[2]);
      printf("[%s]\n", this->executableName);
      if(this->measure_ssd!=NULL)
         printf("[%s] The SSD is used as a similarity measure.\n", this->executableName);
      if(this->measure_kld!=NULL)
         printf("[%s] The KL divergence is used as a similarity measure.\n", this->executableName);
      if(this->measure_lncc!=NULL)
         printf("[%s] The LNCC is used as a similarity measure.\n", this->executableName);
      if(this->measure_dti!=NULL)
         printf("[%s] A DTI based measure is used as a similarity measure.\n", this->executableName);
      if(this->measure_multichannel_nmi!=NULL)
      {
         printf("[%s] The multichannel NMI is used as a similarity measure.\n", this->executableName);
//            if(this->approxParzenWindow)
//                printf("[%s] The Parzen window joint histogram filling is approximated\n", this->executableName);
//            else printf("[%s] The Parzen window joint histogram filling is not approximated\n", this->executableName);
      }
      if(this->measure_nmi!=NULL || (this->measure_dti==NULL && this->measure_kld==NULL &&
                                     this->measure_lncc==NULL && this->measure_multichannel_nmi==NULL &&
                                     this->measure_nmi==NULL && this->measure_ssd==NULL) )
      {
         printf("[%s] The NMI is used as a similarity measure.\n", this->executableName);
//            if(this->approxParzenWindow)
//                printf("[%s] The Parzen window joint histogram filling is approximated\n", this->executableName);
//            else printf("[%s] The Parzen window joint histogram filling is not approximated\n", this->executableName);
      }
      printf("[%s] Similarity measure term weight: %g\n", this->executableName, this->similarityWeight);
      printf("[%s]\n", this->executableName);
      printf("[%s] Bending energy penalty term weight: %g\n", this->executableName, this->bendingEnergyWeight);
      printf("[%s]\n", this->executableName);
      printf("[%s] Linear energy penalty term weights: %g %g\n", this->executableName,
             this->linearEnergyWeight0, this->linearEnergyWeight1);
      printf("[%s]\n", this->executableName);
      printf("[%s] L2 norm of the displacement penalty term weights: %g\n", this->executableName,
             this->L2NormWeight);
      printf("[%s]\n", this->executableName);
      printf("[%s] Jacobian-based penalty term weight: %g\n", this->executableName, this->jacobianLogWeight);
      if(this->jacobianLogWeight>0)
      {
         if(this->jacobianLogApproximation) printf("[%s] \t* Jacobian-based penalty term is approximated\n",
                  this->executableName);
         else printf("[%s] \t* Jacobian-based penalty term is not approximated\n", this->executableName);
      }
#ifdef NDEBUG
   }
#endif

   this->initialised=true;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::Initialise");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetDeformationField()
{
   reg_spline_getDeformationField(this->controlPointGrid,
                                  this->deformationFieldImage,
                                  this->currentMask,
                                  false, //composition
                                  true // bspline
                                 );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetDeformationField");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
   if(this->jacobianLogWeight<=0) return 0.;

   double value=0.;

   if(type==2)
   {
      value = reg_spline_getJacobianPenaltyTerm(this->controlPointGrid,
              this->currentReference,
              false);
   }
   else
   {
      value = reg_spline_getJacobianPenaltyTerm(this->controlPointGrid,
              this->currentReference,
              this->jacobianLogApproximation);
   }
   unsigned int maxit=5;
   if(type>0) maxit=20;
   unsigned int it=0;
   while(value!=value && it<maxit)
   {
      if(type==2)
      {
         value = reg_spline_correctFolding(this->controlPointGrid,
                                           this->currentReference,
                                           false);
      }
      else
      {
         value = reg_spline_correctFolding(this->controlPointGrid,
                                           this->currentReference,
                                           this->jacobianLogApproximation);
      }
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] Folding correction\n");
#endif
      it++;
   }
   if(type>0)
   {
      if(value!=value)
      {
         this->optimiser->RestoreBestDOF();
         fprintf(stderr, "[NiftyReg ERROR] The folding correction scheme failed\n");
      }
      else
      {
#ifndef NDEBUG
         if(it>0)
            printf("[%s] Folding correction, %i step(s)\n", this->executableName, it);
#endif
      }
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeJacobianBasedPenaltyTerm");
#endif
   return (double)this->jacobianLogWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeBendingEnergyPenaltyTerm()
{
   if(this->bendingEnergyWeight<=0) return 0.;

   double value = reg_spline_approxBendingEnergy(this->controlPointGrid);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeBendingEnergyPenaltyTerm");
#endif
   return this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeLinearEnergyPenaltyTerm()
{
   if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0)
      return 0.;

   double values_le[2]= {0.,0.};
   reg_spline_linearEnergy(this->controlPointGrid, values_le);

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeLinearEnergyPenaltyTerm");
#endif
   return this->linearEnergyWeight0*values_le[0] +
          this->linearEnergyWeight1*values_le[1];
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::ComputeL2NormDispPenaltyTerm()
{
   if(this->L2NormWeight<=0)
      return 0.;

   double values_l2=reg_spline_L2norm_displacement(this->controlPointGrid);

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::ComputeL2NormDispPenaltyTerm");
#endif
   return (double)this->L2NormWeight*values_l2;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetSimilarityMeasureGradient()
{
   this->GetVoxelBasedGradient();

   // The voxel based NMI gradient is convolved with a spline kernel
   // Convolution along the x axis
   float currentNodeSpacing[3];
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dx;
   bool activeAxis[3]= {1,0,0};
   reg_tools_kernelConvolution(this->voxelBasedMeasureGradientImage,
                               currentNodeSpacing,
                               1, // cubic spline kernel
                               NULL, // mask
                               NULL, // all volumes are considered as active
                               activeAxis
                              );
   // Convolution along the y axis
   currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dy;
   activeAxis[0]=0;
   activeAxis[1]=1;
   reg_tools_kernelConvolution(this->voxelBasedMeasureGradientImage,
                               currentNodeSpacing,
                               1, // cubic spline kernel
                               NULL, // mask
                               NULL, // all volumes are considered as active
                               activeAxis
                              );
   // Convolution along the z axis if required
   if(this->voxelBasedMeasureGradientImage->nz>1)
   {
      currentNodeSpacing[0]=currentNodeSpacing[1]=currentNodeSpacing[2]=this->controlPointGrid->dz;
      activeAxis[1]=0;
      activeAxis[2]=1;
      reg_tools_kernelConvolution(this->voxelBasedMeasureGradientImage,
                                  currentNodeSpacing,
                                  1, // cubic spline kernel
                                  NULL, // mask
                                  NULL, // all volumes are considered as active
                                  activeAxis
                                 );
   }

   // The node based NMI gradient is extracted
   mat44 reorientation;
   if(this->currentFloating->sform_code>0)
      reorientation = this->currentFloating->sto_ijk;
   else reorientation = this->currentFloating->qto_ijk;
   reg_voxelCentric2NodeCentric(this->transformationGradient,
                                this->voxelBasedMeasureGradientImage,
                                this->similarityWeight,
                                false, // no update
                                &reorientation
                               );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetSimilarityMeasureGradient");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetBendingEnergyGradient()
{
   if(this->bendingEnergyWeight<=0) return;

   reg_spline_approxBendingEnergyGradient(this->controlPointGrid,
                                          this->transformationGradient,
                                          this->bendingEnergyWeight);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetBendingEnergyGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetLinearEnergyGradient()
{
   if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0) return;

   reg_spline_linearEnergyGradient(this->controlPointGrid,
                                   this->currentReference,
                                   this->transformationGradient,
                                   this->linearEnergyWeight0,
                                   this->linearEnergyWeight1);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetLinearEnergyGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetL2NormDispGradient()
{
   if(this->L2NormWeight<=0) return;

   reg_spline_L2norm_dispGradient(this->controlPointGrid,
                                  this->currentReference,
                                  this->transformationGradient,
                                  this->L2NormWeight);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetL2NormDispGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetJacobianBasedGradient()
{
   if(this->jacobianLogWeight<=0) return;

   reg_spline_getJacobianPenaltyTermGradient(this->controlPointGrid,
         this->currentReference,
         this->transformationGradient,
         this->jacobianLogWeight,
         this->jacobianLogApproximation);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetJacobianBasedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::SetGradientImageToZero()
{
   T* nodeGradPtr = static_cast<T *>(this->transformationGradient->data);
   for(size_t i=0; i<this->transformationGradient->nvox; ++i)
      *nodeGradPtr++=0;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetGradientImageToZero");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d<T>::NormaliseGradient()
{
   // First compute the gradient max length for normalisation purpose
//	T maxGradValue=0;
   size_t voxNumber = this->transformationGradient->nx *
                      this->transformationGradient->ny *
                      this->transformationGradient->nz;
   T *ptrX = static_cast<T *>(this->transformationGradient->data);
   T *ptrY = &ptrX[voxNumber];
   T *ptrZ = NULL;
   T maxGradValue=0;
//	float *length=(float *)calloc(voxNumber,sizeof(float));
   if(this->transformationGradient->nz>1)
   {
      ptrZ = &ptrY[voxNumber];
      for(size_t i=0; i<voxNumber; i++)
      {
         T valX=0,valY=0,valZ=0;
         if(this->optimiseX==true)
            valX = *ptrX++;
         if(this->optimiseY==true)
            valY = *ptrY++;
         if(this->optimiseZ==true)
            valZ = *ptrZ++;
//			length[i] = (float)(sqrt(valX*valX + valY*valY + valZ*valZ));
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
            valX = *ptrX++;
         if(this->optimiseY==true)
            valY = *ptrY++;
//			length[i] = (float)(sqrt(valX*valX + valY*valY));
         T length = (T)(sqrt(valX*valX + valY*valY));
         maxGradValue = (length>maxGradValue)?length:maxGradValue;
      }
   }
//	reg_heapSort(length,voxNumber);
//	T maxGradValue = (T)(length[90*voxNumber/100 - 1]);
//	free(length);


   if(strcmp(this->executableName,"NiftyReg F3D")==0)
   {
      // The gradient is normalised if we are running f3d
      // It will be normalised later when running f3d_sym or f3d2
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] Objective function gradient maximal length: %g\n",maxGradValue);
#endif
      ptrX = static_cast<T *>(this->transformationGradient->data);
      if(this->transformationGradient->nz>1)
      {
         ptrX = static_cast<T *>(this->transformationGradient->data);
         ptrY = &ptrX[voxNumber];
         ptrZ = &ptrY[voxNumber];
         for(size_t i=0; i<voxNumber; ++i)
         {
            T valX=0,valY=0,valZ=0;
            if(this->optimiseX==true)
               valX = *ptrX;
            if(this->optimiseY==true)
               valY = *ptrY;
            if(this->optimiseZ==true)
               valZ = *ptrZ;
//				T tempLength = (float)(sqrt(valX*valX + valY*valY + valZ*valZ));
//				if(tempLength>maxGradValue){
//					*ptrX *= maxGradValue / tempLength;
//					*ptrY *= maxGradValue / tempLength;
//					*ptrZ *= maxGradValue / tempLength;
//				}
            *ptrX++ = valX / maxGradValue;
            *ptrY++ = valY / maxGradValue;
            *ptrZ++ = valZ / maxGradValue;
         }
      }
      else
      {
         ptrX = static_cast<T *>(this->transformationGradient->data);
         ptrY = &ptrX[voxNumber];
         for(size_t i=0; i<voxNumber; ++i)
         {
            T valX=0,valY=0;
            if(this->optimiseX==true)
               valX = *ptrX;
            if(this->optimiseY==true)
               valY = *ptrY;
//				T tempLength = (float)(sqrt(valX*valX + valY*valY));
//				if(tempLength>maxGradValue){
//					*ptrX *= maxGradValue / tempLength;
//					*ptrY *= maxGradValue / tempLength;
//				}
            *ptrX++ = valX / maxGradValue;
            *ptrY++ = valY / maxGradValue;
         }
      }
   }
   // Returns the largest gradient distance
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::NormaliseGradient");
#endif
   return maxGradValue;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::DisplayCurrentLevelParameters()
{
#ifdef NDEBUG
   if(this->verbose)
   {
#endif
      printf("[%s] **************************************************\n", this->executableName);
      printf("[%s] Current level: %i / %i\n", this->executableName, this->currentLevel+1, this->levelNumber);
      printf("[%s] Current reference image\n", this->executableName);
      printf("[%s] \t* image dimension: %i x %i x %i x %i\n", this->executableName,
             this->currentReference->nx, this->currentReference->ny,
             this->currentReference->nz,this->currentReference->nt);
      printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
             this->currentReference->dx, this->currentReference->dy,
             this->currentReference->dz);
      printf("[%s] Current floating image\n", this->executableName);
      printf("[%s] \t* image dimension: %i x %i x %i x %i\n", this->executableName,
             this->currentFloating->nx, this->currentFloating->ny,
             this->currentFloating->nz,this->currentFloating->nt);
      printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
             this->currentFloating->dx, this->currentFloating->dy,
             this->currentFloating->dz);
      printf("[%s] Current control point image\n", this->executableName);
      printf("[%s] \t* image dimension: %i x %i x %i\n", this->executableName,
             this->controlPointGrid->nx, this->controlPointGrid->ny,
             this->controlPointGrid->nz);
      printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
             this->controlPointGrid->dx, this->controlPointGrid->dy,
             this->controlPointGrid->dz);
#ifdef NDEBUG
   }
#endif

#ifndef NDEBUG
   if(this->currentReference->sform_code>0)
      reg_mat44_disp(&(this->currentReference->sto_xyz), (char *)"[NiftyReg DEBUG] Reference sform");
   else reg_mat44_disp(&(this->currentReference->qto_xyz), (char *)"[NiftyReg DEBUG] Reference qform");

   if(this->currentFloating->sform_code>0)
      reg_mat44_disp(&(this->currentFloating->sto_xyz), (char *)"[NiftyReg DEBUG] Floating sform");
   else reg_mat44_disp(&(this->currentFloating->qto_xyz), (char *)"[NiftyReg DEBUG] Floating qform");

   if(this->controlPointGrid->sform_code>0)
      reg_mat44_disp(&(this->controlPointGrid->sto_xyz), (char *)"[NiftyReg DEBUG] CPP sform");
   else reg_mat44_disp(&(this->controlPointGrid->qto_xyz), (char *)"[NiftyReg DEBUG] CPP qform");
#endif
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::DisplayCurrentLevelParameters");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d<T>::GetObjectiveFunctionValue()
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
   else
   {
      reg_print_msg_warn("No measure of similarity is part of the cost function");
   }
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] (wMeasure) %g | (wBE) %g | (wLE) %g | (wL2) %g | (wJac) %g\n",
          this->currentWMeasure,
          this->currentWBE,
          this->currentWLE,
          this->currentWL2,
          this->currentWJac);
#endif

#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionValue");
#endif
   // Store the global objective function value
   return this->currentWMeasure - this->currentWBE - this->currentWLE - this->currentWL2 - this->currentWJac;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::UpdateParameters(float scale)
{
   T *currentDOF=this->optimiser->GetCurrentDOF();
   T *bestDOF=this->optimiser->GetBestDOF();
   T *gradient=this->optimiser->GetGradient();

   // Update the control point position
   if(this->optimiser->GetOptimiseX()==true &&
         this->optimiser->GetOptimiseY()==true &&
         this->optimiser->GetOptimiseZ()==true)
   {
      // Update the values for all axis displacement
      for(size_t i=0; i<this->optimiser->GetDOFNumber(); ++i)
      {
         currentDOF[i] = bestDOF[i] + scale * gradient[i];
      }
   }
   else
   {
      size_t voxNumber = this->optimiser->GetVoxNumber();
      // Update the values for the x-axis displacement
      if(this->optimiser->GetOptimiseX()==true)
      {
         for(size_t i=0; i<voxNumber; ++i)
         {
            currentDOF[i] = bestDOF[i] + scale * gradient[i];
         }
      }
      // Update the values for the y-axis displacement
      if(this->optimiser->GetOptimiseY()==true)
      {
         T *currentDOFY=&currentDOF[voxNumber];
         T *bestDOFY=&bestDOF[voxNumber];
         T *gradientY=&gradient[voxNumber];
         for(size_t i=0; i<voxNumber; ++i)
         {
            currentDOFY[i] = bestDOFY[i] + scale * gradientY[i];
         }
      }
      // Update the values for the z-axis displacement
      if(this->optimiser->GetOptimiseZ()==true && this->optimiser->GetNDim()>2)
      {
         T *currentDOFZ=&currentDOF[2*voxNumber];
         T *bestDOFZ=&bestDOF[2*voxNumber];
         T *gradientZ=&gradient[2*voxNumber];
         for(size_t i=0; i<voxNumber; ++i)
         {
            currentDOFZ[i] = bestDOFZ[i] + scale * gradientZ[i];
         }
      }
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::UpdateParameters");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::SetOptimiser()
{
   reg_base<T>::SetOptimiser();
   this->optimiser->Initialise(this->controlPointGrid->nvox,
                               this->controlPointGrid->nz>1?3:2,
                               this->optimiseX,
                               this->optimiseY,
                               this->optimiseZ,
                               this->maxiterationNumber,
                               0, // currentIterationNumber,
                               this,
                               static_cast<T *>(this->controlPointGrid->data),
                               static_cast<T *>(this->transformationGradient->data)
                              );
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SetOptimiser");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::SmoothGradient()
{
   // The gradient is smoothed using a Gaussian kernel if it is required
   if(this->gradientSmoothingSigma!=0)
   {
      float kernel = fabs(this->gradientSmoothingSigma);
      reg_tools_kernelConvolution(this->transformationGradient,
                                  &kernel,
                                  0);
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::SmoothGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d<T>::GetApproximatedGradient()
{
   // Loop over every control point
   T *gridPtr = static_cast<T *>(this->controlPointGrid->data);
   T *gradPtr = static_cast<T *>(this->transformationGradient->data);
   T eps = this->controlPointGrid->dx / 100.f;
   for(size_t i=0; i<this->controlPointGrid->nvox; ++i)
   {
      T currentValue = this->optimiser->GetBestDOF()[i];
      gridPtr[i] = currentValue + eps;
      double valPlus = this->GetObjectiveFunctionValue();
      gridPtr[i] = currentValue - eps;
      double valMinus = this->GetObjectiveFunctionValue();
      gridPtr[i] = currentValue;
      gradPtr[i] = -(T)((valPlus - valMinus ) / (2.0*eps));
   }
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetApproximatedGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image **reg_f3d<T>::GetWarpedImage()
{
   // The initial images are used
   if(this->inputReference==NULL ||
         this->inputFloating==NULL ||
         this->controlPointGrid==NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] reg_f3d::GetWarpedImage()\n");
      fprintf(stderr," * The reference, floating and control point grid images have to be defined\n");
   }

   this->currentReference = this->inputReference;
   this->currentFloating = this->inputFloating;
   this->currentMask=NULL;

   reg_base<T>::AllocateWarped();
   reg_base<T>::AllocateDeformationField();
   reg_base<T>::WarpFloatingImage(3); // cubic spline interpolation
   reg_base<T>::ClearDeformationField();

   nifti_image **resultImage= (nifti_image **)malloc(2*sizeof(nifti_image *));
   resultImage[0]=nifti_copy_nim_info(this->warped);
   resultImage[0]->cal_min=this->inputFloating->cal_min;
   resultImage[0]->cal_max=this->inputFloating->cal_max;
   resultImage[0]->scl_slope=this->inputFloating->scl_slope;
   resultImage[0]->scl_inter=this->inputFloating->scl_inter;
   resultImage[0]->data=(void *)malloc(resultImage[0]->nvox*resultImage[0]->nbyper);
   memcpy(resultImage[0]->data, this->warped->data, resultImage[0]->nvox*resultImage[0]->nbyper);

   resultImage[1]=NULL;

   reg_f3d<T>::ClearWarped();
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetWarpedImage");
#endif
   return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image * reg_f3d<T>::GetControlPointPositionImage()
{
   nifti_image *returnedControlPointGrid = nifti_copy_nim_info(this->controlPointGrid);
   returnedControlPointGrid->data=(void *)malloc(returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
   memcpy(returnedControlPointGrid->data, this->controlPointGrid->data,
          returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
   return returnedControlPointGrid;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetControlPointPositionImage");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::UpdateBestObjFunctionValue()
{
   this->bestWMeasure=this->currentWMeasure;
   this->bestWBE=this->currentWBE;
   this->bestWLE=this->currentWLE;
   this->bestWL2=this->currentWL2;
   this->bestWJac=this->currentWJac;
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::UpdateBestObjFunctionValue");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::PrintInitialObjFunctionValue()
{
   if(!this->verbose) return;

   double bestValue=this->optimiser->GetBestObjFunctionValue();

   printf("[%s] Initial objective function: %g = (wSIM)%g - (wBE)%g - (wLE)%g - (wL2)%g - (wJAC)%g\n",
          this->executableName, bestValue, this->bestWMeasure, this->bestWBE, this->bestWLE, this->bestWL2, this->bestWJac);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::PrintInitialObjFunctionValue");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::PrintCurrentObjFunctionValue(T currentSize)
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
   printf(" [+ %g mm]\n", currentSize);
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::PrintCurrentObjFunctionValue");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::GetObjectiveFunctionGradient()
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

   if(!this->useApproxGradient)
   {
      // Compute the penalty term gradients if required
      this->GetBendingEnergyGradient();
      this->GetJacobianBasedGradient();
      this->GetLinearEnergyGradient();
      this->GetL2NormDispGradient();
   }
   else this->GetApproximatedGradient();

   this->optimiser->IncrementCurrentIterationNumber();

   // Smooth the gradient if require
   this->SmoothGradient();
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::GetObjectiveFunctionGradient");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d<T>::CorrectTransformation()
{
   if(this->jacobianLogWeight>0 && this->jacobianLogApproximation==true)
      this->ComputeJacobianBasedPenaltyTerm(2); // 20 iterations without approximation
#ifndef NDEBUG
   reg_print_fct_debug("reg_f3d<T>::CorrectTransformation");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif
