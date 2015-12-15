/*
 *  _reg_f3d_gpu.cpp
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_GPU_CPP
#define _REG_F3D_GPU_CPP

#include "_reg_f3d_gpu.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_f3d_gpu::reg_f3d_gpu(int refTimePoint,int floTimePoint)
   : reg_f3d<float>::reg_f3d(refTimePoint,floTimePoint)
{
   this->executableName=(char *)"NiftyReg F3D GPU";
   this->currentReference_gpu=NULL;
   this->currentFloating_gpu=NULL;
   this->currentMask_gpu=NULL;
   this->warped_gpu=NULL;
   this->controlPointGrid_gpu=NULL;
   this->deformationFieldImage_gpu=NULL;
   this->warpedGradientImage_gpu=NULL;
   this->voxelBasedMeasureGradientImage_gpu=NULL;
   this->transformationGradient_gpu=NULL;

   this->measure_gpu_ssd=NULL;
   this->measure_gpu_kld=NULL;
   this->measure_gpu_dti=NULL;
   this->measure_gpu_lncc=NULL;
   this->measure_gpu_nmi=NULL;
   this->measure_gpu_multichannel_nmi=NULL;

   this->currentReference2_gpu=NULL;
   this->currentFloating2_gpu=NULL;
   this->warped2_gpu=NULL;
   this->warpedGradientImage2_gpu=NULL;

#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
reg_f3d_gpu::~reg_f3d_gpu()
{
   if(this->currentReference_gpu!=NULL)
      cudaCommon_free(&this->currentReference_gpu);
   if(this->currentFloating_gpu!=NULL)
      cudaCommon_free(&this->currentFloating_gpu);
   if(this->currentMask_gpu!=NULL)
      cudaCommon_free<int>(&this->currentMask_gpu);
   if(this->warped_gpu!=NULL)
      cudaCommon_free<float>(&this->warped_gpu);
   if(this->controlPointGrid_gpu!=NULL)
      cudaCommon_free<float4>(&this->controlPointGrid_gpu);
   if(this->deformationFieldImage_gpu!=NULL)
      cudaCommon_free<float4>(&this->deformationFieldImage_gpu);
   if(this->warpedGradientImage_gpu!=NULL)
      cudaCommon_free<float4>(&this->warpedGradientImage_gpu);
   if(this->voxelBasedMeasureGradientImage_gpu!=NULL)
      cudaCommon_free<float4>(&this->voxelBasedMeasureGradientImage_gpu);
   if(this->transformationGradient_gpu!=NULL)
      cudaCommon_free<float4>(&this->transformationGradient_gpu);

   if(this->currentReference2_gpu!=NULL)
      cudaCommon_free(&this->currentReference2_gpu);
   if(this->currentFloating2_gpu!=NULL)
      cudaCommon_free(&this->currentFloating2_gpu);
   if(this->warped2_gpu!=NULL)
      cudaCommon_free<float>(&this->warped2_gpu);
   if(this->warpedGradientImage2_gpu!=NULL)
      cudaCommon_free<float4>(&this->warpedGradientImage2_gpu);

   if(this->optimiser!=NULL)
   {
      delete this->optimiser;
      this->optimiser=NULL;
   }

   if(this->measure_gpu_nmi!=NULL)
   {
      delete this->measure_gpu_nmi;
      this->measure_gpu_nmi=NULL;
      this->measure_nmi=NULL;
   }
   if(this->measure_gpu_multichannel_nmi!=NULL)
   {
      delete this->measure_gpu_multichannel_nmi;
      this->measure_gpu_multichannel_nmi=NULL;
      this->measure_multichannel_nmi=NULL;
   }
   if(this->measure_gpu_ssd!=NULL)
   {
      delete this->measure_gpu_ssd;
      this->measure_gpu_ssd=NULL;
      this->measure_ssd=NULL;
   }
   if(this->measure_gpu_kld!=NULL)
   {
      delete this->measure_gpu_kld;
      this->measure_gpu_kld=NULL;
      this->measure_kld=NULL;
   }
   if(this->measure_gpu_dti!=NULL)
   {
      delete this->measure_gpu_dti;
      this->measure_gpu_dti=NULL;
      this->measure_dti=NULL;
   }
   if(this->measure_gpu_lncc!=NULL)
   {
      delete this->measure_gpu_lncc;
      this->measure_gpu_lncc=NULL;
      this->measure_lncc=NULL;
   }

   NR_CUDA_SAFE_CALL(cudaThreadExit())
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateWarped()
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateWarped called.\n");
#endif
   if(this->currentReference==NULL)
   {
      printf("[NiftyReg ERROR] Error when allocating the warped image.\n");
      reg_exit(1);
   }
   this->ClearWarped();
   this->warped = nifti_copy_nim_info(this->currentReference);
   this->warped->dim[0]=this->warped->ndim=this->currentFloating->ndim;
   this->warped->dim[4]=this->warped->nt=this->currentFloating->nt;
   this->warped->pixdim[4]=this->warped->dt=1.0;
   this->warped->nvox = this->warped->nx *
                        this->warped->ny *
                        this->warped->nz *
                        this->warped->nt;
   this->warped->datatype = this->currentFloating->datatype;
   this->warped->nbyper = this->currentFloating->nbyper;
   NR_CUDA_SAFE_CALL(cudaMallocHost(&(this->warped->data), this->warped->nvox*this->warped->nbyper))
   if(this->warped->nt==1)
   {
      if(cudaCommon_allocateArrayToDevice<float>(&this->warped_gpu, this->warped->dim))
      {
         printf("[NiftyReg ERROR] Error when allocating the warped image.\n");
         reg_exit(1);
      }
   }
   else if(this->warped->nt==2)
   {
      if(cudaCommon_allocateArrayToDevice<float>(&this->warped_gpu, &this->warped2_gpu, this->warped->dim))
      {
         printf("[NiftyReg ERROR] Error when allocating the warped image.\n");
         reg_exit(1);
      }
   }
   else
   {
      printf("[NiftyReg ERROR] reg_f3d_gpu does not handle more than 2 time points in the floating image.\n");
      reg_exit(1);
   }
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateWarped done.\n");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearWarped()
{
   if(this->warped!=NULL)
   {
      NR_CUDA_SAFE_CALL(cudaFreeHost(this->warped->data))
      this->warped->data = NULL;
      nifti_image_free(this->warped);
      this->warped=NULL;
   }
   if(this->warped_gpu!=NULL)
   {
      cudaCommon_free<float>(&this->warped_gpu);
      this->warped_gpu=NULL;
   }
   if(this->warped2_gpu!=NULL)
   {
      cudaCommon_free<float>(&this->warped2_gpu);
      this->warped2_gpu=NULL;
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateDeformationField()
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateDeformationField called.\n");
#endif
   this->ClearDeformationField();
   NR_CUDA_SAFE_CALL(cudaMalloc(&this->deformationFieldImage_gpu,
                                this->activeVoxelNumber[this->currentLevel]*sizeof(float4)))

#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateDeformationField done.\n");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearDeformationField()
{
   if(this->deformationFieldImage_gpu!=NULL)
   {
      cudaCommon_free<float4>(&this->deformationFieldImage_gpu);
      this->deformationFieldImage_gpu=NULL;
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateWarpedGradient()
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateWarpedGradient called.\n");
#endif
   this->ClearWarpedGradient();
   if(this->inputFloating->nt==1)
   {
      NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage_gpu,
                                   this->activeVoxelNumber[this->currentLevel]*sizeof(float4)))
   }
   else if(this->inputFloating->nt==2)
   {
      NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage_gpu,
                                   this->activeVoxelNumber[this->currentLevel]*sizeof(float4)))
      NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage2_gpu,
                                   this->activeVoxelNumber[this->currentLevel]*sizeof(float4)))
   }
   else
   {
      printf("[NiftyReg ERROR] reg_f3d_gpu does not handle more than 2 time points in the floating image.\n");
      reg_exit(1);
   }
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateWarpedGradient done.\n");
#endif

   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearWarpedGradient()
{
   if(this->warpedGradientImage_gpu!=NULL)
   {
      cudaCommon_free<float4>(&this->warpedGradientImage_gpu);
      this->warpedGradientImage_gpu=NULL;
   }
   if(this->warpedGradientImage2_gpu!=NULL)
   {
      cudaCommon_free<float4>(&this->warpedGradientImage2_gpu);
      this->warpedGradientImage2_gpu=NULL;
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateVoxelBasedMeasureGradient()
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateVoxelBasedMeasureGradient called.\n");
#endif
   this->ClearVoxelBasedMeasureGradient();
   if(cudaCommon_allocateArrayToDevice(&this->voxelBasedMeasureGradientImage_gpu,
                                       this->currentReference->dim))
   {
      printf("[NiftyReg ERROR] Error when allocating the voxel based measure gradient image.\n");
      reg_exit(1);
   }
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateVoxelBasedMeasureGradient done.\n");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearVoxelBasedMeasureGradient()
{
   if(this->voxelBasedMeasureGradientImage_gpu!=NULL)
   {
      cudaCommon_free<float4>(&this->voxelBasedMeasureGradientImage_gpu);
      this->voxelBasedMeasureGradientImage_gpu=NULL;
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::AllocateTransformationGradient()
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateNodeBasedGradient called.\n");
#endif
   this->ClearTransformationGradient();
   if(cudaCommon_allocateArrayToDevice(&this->transformationGradient_gpu,
                                       this->controlPointGrid->dim))
   {
      printf("[NiftyReg ERROR] Error when allocating the node based gradient image.\n");
      reg_exit(1);
   }
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateNodeBasedGradient done.\n");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearTransformationGradient()
{
   if(this->transformationGradient_gpu!=NULL)
   {
      cudaCommon_free<float4>(&this->transformationGradient_gpu);
      this->transformationGradient_gpu=NULL;
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_f3d_gpu::ComputeJacobianBasedPenaltyTerm(int type)
{
   if(this->jacobianLogWeight<=0) return 0.;

   double value;
   if(type==2)
   {
      value = reg_spline_getJacobianPenaltyTerm_gpu(this->currentReference,
              this->controlPointGrid,
              &this->controlPointGrid_gpu,
              false);
   }
   else
   {
      value = reg_spline_getJacobianPenaltyTerm_gpu(this->currentReference,
              this->controlPointGrid,
              &this->controlPointGrid_gpu,
              this->jacobianLogApproximation);
   }
   unsigned int maxit=5;
   if(type>0) maxit=20;
   unsigned int it=0;
   while(value!=value && it<maxit)
   {
      if(type==2)
      {
         value = reg_spline_correctFolding_gpu(this->currentReference,
                                               this->controlPointGrid,
                                               &this->controlPointGrid_gpu,
                                               false);
      }
      else
      {
         value = reg_spline_correctFolding_gpu(this->currentReference,
                                               this->controlPointGrid,
                                               &this->controlPointGrid_gpu,
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
#ifdef NDEBUG
         if(this->verbose)
         {
#endif
            printf("[NiftyReg F3D] Folding correction, %i step(s)\n", it);
#ifdef NDEBUG
         }
#endif
      }
   }
   return (double)this->jacobianLogWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
double reg_f3d_gpu::ComputeBendingEnergyPenaltyTerm()
{
   if(this->bendingEnergyWeight<=0) return 0.;

   double value = reg_spline_approxBendingEnergy_gpu(this->controlPointGrid,
                  &this->controlPointGrid_gpu);
   return this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetDeformationField()
{
   if(this->controlPointGrid_gpu==NULL)
   {
      reg_f3d<float>::GetDeformationField();
   }
   else
   {
      // Compute the deformation field
      reg_spline_getDeformationField_gpu(this->controlPointGrid,
                                         this->currentReference,
                                         &this->controlPointGrid_gpu,
                                         &this->deformationFieldImage_gpu,
                                         &this->currentMask_gpu,
                                         this->activeVoxelNumber[this->currentLevel],
                                         true // use B-splines
                                        );
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::WarpFloatingImage(int inter)
{
   // Interpolation is linear by default when using GPU, the inter variable is not used.
   inter=inter; // just to avoid a compiler warning

   // Compute the deformation field
   this->GetDeformationField();

   // Resample the floating image
   reg_resampleImage_gpu(this->currentFloating,
                         &this->warped_gpu,
                         &this->currentFloating_gpu,
                         &this->deformationFieldImage_gpu,
                         &this->currentMask_gpu,
                         this->activeVoxelNumber[this->currentLevel],
                         this->warpedPaddingValue);
   if(this->currentFloating->nt==2)
   {
      reg_resampleImage_gpu(this->currentFloating,
                            &this->warped2_gpu,
                            &this->currentFloating2_gpu,
                            &this->deformationFieldImage_gpu,
                            &this->currentMask_gpu,
                            this->activeVoxelNumber[this->currentLevel],
                            this->warpedPaddingValue);
   }

   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::SetGradientImageToZero()
{
   cudaMemset(this->transformationGradient_gpu,0,
              this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz*
              sizeof(float4));
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetVoxelBasedGradient()
{
   // The intensity gradient is first computed
   reg_getImageGradient_gpu(this->currentFloating,
                            &this->currentFloating_gpu,
                            &this->deformationFieldImage_gpu,
                            &this->warpedGradientImage_gpu,
                            this->activeVoxelNumber[this->currentLevel],
                            this->warpedPaddingValue);

   // The voxel based gradient image is filled with zeros
   cudaMemset(this->voxelBasedMeasureGradientImage_gpu,0,
              this->currentReference->nx*this->currentReference->ny*this->currentReference->nz*
              sizeof(float4));
   // The gradient of the various measures of similarity are computed
   if(this->measure_gpu_nmi!=NULL)
      this->measure_gpu_nmi->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_gpu_multichannel_nmi!=NULL)
      this->measure_gpu_multichannel_nmi->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_gpu_ssd!=NULL)
      this->measure_gpu_ssd->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_gpu_kld!=NULL)
      this->measure_gpu_kld->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_gpu_lncc!=NULL)
      this->measure_gpu_lncc->GetVoxelBasedSimilarityMeasureGradient();

   if(this->measure_gpu_dti!=NULL)
      this->measure_gpu_dti->GetVoxelBasedSimilarityMeasureGradient();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetSimilarityMeasureGradient()
{

   this->GetVoxelBasedGradient();

   // The voxel based gradient is smoothed
   float smoothingRadius[3]=
   {
      this->controlPointGrid->dx/this->currentReference->dx,
      this->controlPointGrid->dy/this->currentReference->dy,
      this->controlPointGrid->dz/this->currentReference->dz
   };
   reg_smoothImageForCubicSpline_gpu(this->warped,
                                     &this->voxelBasedMeasureGradientImage_gpu,
                                     smoothingRadius);

   // The node gradient is extracted
   reg_voxelCentric2NodeCentric_gpu(this->warped,
                                    this->controlPointGrid,
                                    &this->voxelBasedMeasureGradientImage_gpu,
                                    &this->transformationGradient_gpu,
                                    this->similarityWeight);

   /* The similarity measure gradient is converted from voxel space to real space */
   mat44 *floatingMatrix_xyz=NULL;
   if(this->currentFloating->sform_code>0)
      floatingMatrix_xyz = &(this->currentFloating->sto_xyz);
   else floatingMatrix_xyz = &(this->currentFloating->qto_xyz);
   reg_convertNMIGradientFromVoxelToRealSpace_gpu( floatingMatrix_xyz,
         this->controlPointGrid,
         &this->transformationGradient_gpu);
   // The gradient is smoothed using a Gaussian kernel if it is required
   if(this->gradientSmoothingSigma!=0)
   {
      reg_gaussianSmoothing_gpu(this->controlPointGrid,
                                &this->transformationGradient_gpu,
                                this->gradientSmoothingSigma,
                                NULL);
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetBendingEnergyGradient()
{
   if(this->bendingEnergyWeight<=0) return;

   reg_spline_approxBendingEnergyGradient_gpu(this->controlPointGrid,
         &this->controlPointGrid_gpu,
         &this->transformationGradient_gpu,
         this->bendingEnergyWeight);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetJacobianBasedGradient()
{
   if(this->jacobianLogWeight<=0) return;

   reg_spline_getJacobianPenaltyTermGradient_gpu(this->currentReference,
         this->controlPointGrid,
         &this->controlPointGrid_gpu,
         &this->transformationGradient_gpu,
         this->jacobianLogWeight,
         this->jacobianLogApproximation);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UpdateParameters(float scale)
{

   float4 *currentDOF=reinterpret_cast<float4 *>(this->optimiser->GetCurrentDOF());
   float4 *bestDOF=reinterpret_cast<float4 *>(this->optimiser->GetBestDOF());
   float4 *gradient=reinterpret_cast<float4 *>(this->optimiser->GetGradient());

   reg_updateControlPointPosition_gpu(this->controlPointGrid,
                                      &currentDOF,
                                      &bestDOF,
                                      &gradient,
                                      scale);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::GetApproximatedGradient()
{
   float4 *gridValue=NULL;
   float4 *modifiedValue=NULL;
   float4 *gradientValue=NULL;
   cudaMallocHost(&gridValue,sizeof(float4));
   cudaMallocHost(&modifiedValue,sizeof(float4));
   cudaMallocHost(&gradientValue,sizeof(float4));

   float eps = this->controlPointGrid->dx / 1000.f;

   for(size_t i=0; i<this->optimiser->GetVoxNumber(); ++i)
   {
      // Extract the current value
      cudaMemcpy(gridValue,
                 &this->controlPointGrid_gpu[i],
                 sizeof(float4),
                 cudaMemcpyDeviceToHost);
      modifiedValue[0]=gridValue[0];
      // -- X axis
      // Modify the current value along the x axis
      modifiedValue[0].x = gridValue[0].x + eps;
      cudaMemcpy(&this->controlPointGrid_gpu[i],
                 modifiedValue,
                 sizeof(float4),
                 cudaMemcpyHostToDevice);
      // Evaluate the objective function value
      gradientValue[0].x=this->GetObjectiveFunctionValue();
      // Modify the current value along the x axis
      modifiedValue[0].x = gridValue[0].x - eps;
      cudaMemcpy(&this->controlPointGrid_gpu[i],
                 modifiedValue,
                 sizeof(float4),
                 cudaMemcpyHostToDevice);
      // Evaluate the objective function value
      gradientValue[0].x -= this->GetObjectiveFunctionValue();
      gradientValue[0].x /= 2.f*eps;
      modifiedValue[0].x = gridValue[0].x;
      // -- Y axis
      // Modify the current value along the y axis
      modifiedValue[0].y = gridValue[0].y + eps;
      cudaMemcpy(&this->controlPointGrid_gpu[i],
                 modifiedValue,
                 sizeof(float4),
                 cudaMemcpyHostToDevice);
      // Evaluate the objective function value
      gradientValue[0].y=this->GetObjectiveFunctionValue();
      // Modify the current value the y axis
      modifiedValue[0].y = gridValue[0].y - eps;
      cudaMemcpy(&this->controlPointGrid_gpu[i],
                 modifiedValue,
                 sizeof(float4),
                 cudaMemcpyHostToDevice);
      // Evaluate the objective function value
      gradientValue[0].y -= this->GetObjectiveFunctionValue();
      gradientValue[0].y /= 2.f*eps;
      modifiedValue[0].y = gridValue[0].y;
      if(this->optimiser->GetNDim()>2)
      {
         // -- Z axis
         // Modify the current value along the y axis
         modifiedValue[0].z = gridValue[0].z + eps;
         cudaMemcpy(&this->controlPointGrid_gpu[i],
                    modifiedValue,
                    sizeof(float4),
                    cudaMemcpyHostToDevice);
         // Evaluate the objective function value
         gradientValue[0].z=this->GetObjectiveFunctionValue();
         // Modify the current value the y axis
         modifiedValue[0].z = gridValue[0].z - eps;
         cudaMemcpy(&this->controlPointGrid_gpu[i],
                    modifiedValue,
                    sizeof(float4),
                    cudaMemcpyHostToDevice);
         // Evaluate the objective function value
         gradientValue[0].z -= this->GetObjectiveFunctionValue();
         gradientValue[0].z /= 2.f*eps;
      }
      // Restore the initial parametrisation
      cudaMemcpy(&this->controlPointGrid_gpu[i],
                 gridValue,
                 sizeof(float4),
                 cudaMemcpyHostToDevice);

      // Save the assessed gradient
      cudaMemcpy(&this->transformationGradient_gpu[i],
                 gradientValue,
                 sizeof(float4),
                 cudaMemcpyHostToDevice);
   }
   cudaFreeHost(gridValue);
   cudaFreeHost(modifiedValue);
   cudaFreeHost(gradientValue);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
float reg_f3d_gpu::InitialiseCurrentLevel()
{
   float maxStepSize=reg_f3d<float>::InitialiseCurrentLevel();

#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateCurrentInputImage called.\n");
#endif

   if(this->currentReference_gpu!=NULL) cudaCommon_free(&this->currentReference_gpu);
   if(this->currentReference2_gpu!=NULL) cudaCommon_free(&this->currentReference2_gpu);
   if(this->currentReference->nt==1)
   {
      if(cudaCommon_allocateArrayToDevice<float>
            (&this->currentReference_gpu, this->currentReference->dim))
      {
         printf("[NiftyReg ERROR] Error when allocating the reference image.\n");
         reg_exit(1);
      }
      if(cudaCommon_transferNiftiToArrayOnDevice<float>
            (&this->currentReference_gpu, this->currentReference))
      {
         printf("[NiftyReg ERROR] Error when transfering the reference image.\n");
         reg_exit(1);
      }
   }
   else if(this->currentReference->nt==2)
   {
      if(cudaCommon_allocateArrayToDevice<float>
            (&this->currentReference_gpu,&this->currentReference2_gpu, this->currentReference->dim))
      {
         printf("[NiftyReg ERROR] Error when allocating the reference image.\n");
         reg_exit(1);
      }
      if(cudaCommon_transferNiftiToArrayOnDevice<float>
            (&this->currentReference_gpu, &this->currentReference2_gpu, this->currentReference))
      {
         printf("[NiftyReg ERROR] Error when transfering the reference image.\n");
         reg_exit(1);
      }
   }

   if(this->currentFloating_gpu!=NULL) cudaCommon_free(&this->currentFloating_gpu);
   if(this->currentFloating2_gpu!=NULL) cudaCommon_free(&this->currentFloating2_gpu);
   if(this->currentReference->nt==1)
   {
      if(cudaCommon_allocateArrayToDevice<float>
            (&this->currentFloating_gpu, this->currentFloating->dim))
      {
         printf("[NiftyReg ERROR] Error when allocating the floating image.\n");
         reg_exit(1);
      }
      if(cudaCommon_transferNiftiToArrayOnDevice<float>
            (&this->currentFloating_gpu, this->currentFloating))
      {
         printf("[NiftyReg ERROR] Error when transfering the floating image.\n");
         reg_exit(1);
      }
   }
   else if(this->currentReference->nt==2)
   {
      if(cudaCommon_allocateArrayToDevice<float>
            (&this->currentFloating_gpu, &this->currentFloating2_gpu, this->currentFloating->dim))
      {
         printf("[NiftyReg ERROR] Error when allocating the floating image.\n");
         reg_exit(1);
      }
      if(cudaCommon_transferNiftiToArrayOnDevice<float>
            (&this->currentFloating_gpu, &this->currentFloating2_gpu, this->currentFloating))
      {
         printf("[NiftyReg ERROR] Error when transfering the floating image.\n");
         reg_exit(1);
      }
   }
   if(this->controlPointGrid_gpu!=NULL) cudaCommon_free<float4>(&this->controlPointGrid_gpu);
   if(cudaCommon_allocateArrayToDevice<float4>
         (&this->controlPointGrid_gpu, this->controlPointGrid->dim))
   {
      printf("[NiftyReg ERROR] Error when allocating the control point image.\n");
      reg_exit(1);
   }

   if(cudaCommon_transferNiftiToArrayOnDevice<float4>
         (&this->controlPointGrid_gpu, this->controlPointGrid))
   {
      printf("[NiftyReg ERROR] Error when transfering the control point image.\n");
      reg_exit(1);
   }

   int *targetMask_h;
   NR_CUDA_SAFE_CALL(cudaMallocHost(&targetMask_h,this->activeVoxelNumber[this->currentLevel]*sizeof(int)))
   int *targetMask_h_ptr = &targetMask_h[0];
   for(int i=0; i<this->currentReference->nx*this->currentReference->ny*this->currentReference->nz; i++)
   {
      if( this->currentMask[i]!=-1) *targetMask_h_ptr++=i;
   }
   NR_CUDA_SAFE_CALL(cudaMalloc(&this->currentMask_gpu,
                                this->activeVoxelNumber[this->currentLevel]*sizeof(int)))
   NR_CUDA_SAFE_CALL(cudaMemcpy(this->currentMask_gpu, targetMask_h,
                                this->activeVoxelNumber[this->currentLevel]*sizeof(int),
                                cudaMemcpyHostToDevice))
   NR_CUDA_SAFE_CALL(cudaFreeHost(targetMask_h))
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::AllocateCurrentInputImage done.\n");
#endif
   return maxStepSize;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::ClearCurrentInputImage()
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::ClearCurrentInputImage called.\n");
#endif
   if(cudaCommon_transferFromDeviceToNifti<float4>
         (this->controlPointGrid, &this->controlPointGrid_gpu))
   {
      printf("[NiftyReg ERROR] Error when transfering back the control point image.\n");
      reg_exit(1);
   }
   cudaCommon_free<float4>(&this->controlPointGrid_gpu);
   this->controlPointGrid_gpu=NULL;
   cudaCommon_free(&this->currentReference_gpu);
   this->currentReference_gpu=NULL;
   cudaCommon_free(&this->currentFloating_gpu);
   this->currentFloating_gpu=NULL;
   NR_CUDA_SAFE_CALL(cudaFree(this->currentMask_gpu))
   this->currentMask_gpu=NULL;

   if(this->currentReference2_gpu!=NULL)
      cudaCommon_free(&this->currentReference2_gpu);
   this->currentReference2_gpu=NULL;
   if(this->currentFloating2_gpu!=NULL)
      cudaCommon_free(&this->currentFloating2_gpu);
   this->currentFloating2_gpu=NULL;

   this->currentReference=NULL;
   this->currentMask=NULL;
   this->currentFloating=NULL;
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::ClearCurrentInputImage done.\n");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::SetOptimiser()
{
   if(this->useConjGradient)
      this->optimiser=new reg_conjugateGradient_gpu();
   else this->optimiser=new reg_optimiser_gpu();
   // The cpp and grad images are converted to float * instead of float4
   // to enable compatibility with cpu class
   this->optimiser->Initialise(this->controlPointGrid->nvox,
                               this->controlPointGrid->nz>1?3:2,
                               this->optimiseX,
                               this->optimiseY,
                               this->optimiseZ,
                               this->maxiterationNumber,
                               0, // currentIterationNumber,
                               this,
                               reinterpret_cast<float *>(this->controlPointGrid_gpu),
                               reinterpret_cast<float *>(this->transformationGradient_gpu)
                              );
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
float reg_f3d_gpu::NormaliseGradient()
{
   // First compute the gradient max length for normalisation purpose
   float length = reg_getMaximalLength_gpu(&this->transformationGradient_gpu,
                                           this->optimiser->GetVoxNumber()
                                          );

   if(strcmp(this->executableName,"NiftyReg F3D GPU")==0)
   {
      // The gradient is normalised if we are running F3D
      // It will be normalised later when running symmetric or F3D2
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] Objective function gradient_gpu maximal length: %g\n", length);
#endif
      reg_multiplyValue_gpu(this->optimiser->GetVoxNumber(),
                            &this->transformationGradient_gpu,
                            1.f/length);

   }
   // Returns the largest gradient distance
   return length;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
int reg_f3d_gpu::CheckMemoryMB()
{
   if(!this->initialised) reg_f3d<float>::Initialise();

   size_t referenceVoxelNumber=this->referencePyramid[this->levelToPerform-1]->nx *
                               this->referencePyramid[this->levelToPerform-1]->ny *
                               this->referencePyramid[this->levelToPerform-1]->nz;

   size_t warpedVoxelNumber=this->referencePyramid[this->levelToPerform-1]->nx *
                            this->referencePyramid[this->levelToPerform-1]->ny *
                            this->referencePyramid[this->levelToPerform-1]->nz *
                            this->floatingPyramid[this->levelToPerform-1]->nt ;

   size_t totalMemoryRequiered=0;
   // reference image
   totalMemoryRequiered += this->referencePyramid[this->levelToPerform-1]->nvox * sizeof(float);

   // floating image
   totalMemoryRequiered += this->floatingPyramid[this->levelToPerform-1]->nvox * sizeof(float);

   // warped image
   totalMemoryRequiered += warpedVoxelNumber * sizeof(float);

   // mask image
   totalMemoryRequiered += this->activeVoxelNumber[this->levelToPerform-1] * sizeof(int);

   // deformation field
   totalMemoryRequiered += referenceVoxelNumber * sizeof(float4);

   // voxel based intensity gradient
   totalMemoryRequiered += referenceVoxelNumber * sizeof(float4);

   // voxel based NMI gradient + smoothing
   totalMemoryRequiered += 2 * referenceVoxelNumber * sizeof(float4);

   // control point grid
   size_t cp=1;
   cp *= (int)floor(this->referencePyramid[this->levelToPerform-1]->nx*
                    this->referencePyramid[this->levelToPerform-1]->dx/
                    this->spacing[0])+5;
   cp *= (int)floor(this->referencePyramid[this->levelToPerform-1]->ny*
                    this->referencePyramid[this->levelToPerform-1]->dy/
                    this->spacing[1])+5;
   if(this->referencePyramid[this->levelToPerform-1]->nz>1)
      cp *= (int)floor(this->referencePyramid[this->levelToPerform-1]->nz*
                       this->referencePyramid[this->levelToPerform-1]->dz/
                       this->spacing[2])+5;
   totalMemoryRequiered += cp * sizeof(float4);

   // node based NMI gradient
   totalMemoryRequiered += cp * sizeof(float4);

   // conjugate gradient
   totalMemoryRequiered += 2 * cp * sizeof(float4);


   // HERE TODO

   // jacobian array
   if(this->jacobianLogWeight>0)
      totalMemoryRequiered += 10 * referenceVoxelNumber *
                              sizeof(float);

   return (int)(ceil((float)totalMemoryRequiered/float(1024*1024)));

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseNMISetFloatingBinNumber(int timepoint, int floBinNumber)
{
   if(this->measure_gpu_nmi==NULL)
      this->measure_gpu_nmi=new reg_nmi_gpu;
   this->measure_gpu_nmi->SetActiveTimepoint(timepoint);
   // I am here adding 4 to the specified bin number to accomodate for
   // the spline support
   this->measure_gpu_nmi->SetFloatingBinNumber(floBinNumber+4, timepoint);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseNMISetReferenceBinNumber(int timepoint, int refBinNumber)
{
   if(this->measure_gpu_nmi==NULL)
      this->measure_gpu_nmi=new reg_nmi_gpu;
   this->measure_gpu_nmi->SetActiveTimepoint(timepoint);
   // I am here adding 4 to the specified bin number to accomodate for
   // the spline support
   this->measure_gpu_nmi->SetReferenceBinNumber(refBinNumber+4, timepoint);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseMultiChannelNMI(int timepointNumber, int *timepoint)
{
   if(this->measure_gpu_multichannel_nmi==NULL)
      this->measure_gpu_multichannel_nmi=new reg_multichannel_nmi_gpu;
   for(int i=0; i<timepointNumber; ++i)
      this->measure_gpu_multichannel_nmi->SetActiveTimepoint(timepoint[i]);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseSSD(int timepoint)
{
   if(this->measure_gpu_ssd==NULL)
      this->measure_gpu_ssd=new reg_ssd_gpu;
   this->measure_gpu_ssd->SetActiveTimepoint(timepoint);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseKLDivergence(int timepoint)
{
   if(this->measure_gpu_kld==NULL)
      this->measure_gpu_kld=new reg_kld_gpu;
   this->measure_gpu_kld->SetActiveTimepoint(timepoint);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseLNCC(int timepoint, float stddev)
{
   if(this->measure_gpu_lncc==NULL)
      this->measure_gpu_lncc=new reg_lncc_gpu;
   this->measure_gpu_lncc->SetActiveTimepoint(timepoint);
   this->measure_gpu_lncc->SetKernelStandardDeviation(timepoint,stddev);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::UseDTI(int timepoint[6])
{
   if(this->measure_gpu_dti==NULL)
      this->measure_gpu_dti=new reg_dti_gpu;
   for(int i=0; i<6; ++i)
      this->measure_gpu_dti->SetActiveTimepoint(timepoint[i]);
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_f3d_gpu::InitialiseSimilarity()
{
   // SET THE DEFAULT MEASURE OF SIMILARITY IF NONE HAS BEEN SET
   if(this->measure_gpu_nmi==NULL &&
         this->measure_gpu_ssd==NULL &&
         this->measure_gpu_dti==NULL &&
         this->measure_gpu_kld==NULL &&
         this->measure_gpu_lncc==NULL)
   {
      measure_gpu_nmi=new reg_nmi_gpu;
      for(int i=0; i<this->inputReference->nt; ++i)
         measure_gpu_nmi->SetActiveTimepoint(i);
   }
   if(this->measure_gpu_nmi!=NULL)
   {
      this->measure_gpu_nmi->InitialiseMeasure(this->currentReference,
            this->currentFloating,
            this->currentMask,
            this->activeVoxelNumber[this->currentLevel],
            this->warped,
            this->warpedGradientImage,
            this->voxelBasedMeasureGradientImage,
            &this->currentReference_gpu,
            &this->currentFloating_gpu,
            &this->currentMask_gpu,
            &this->warped_gpu,
            &this->warpedGradientImage_gpu,
            &this->voxelBasedMeasureGradientImage_gpu
                                              );
      this->measure_nmi=this->measure_gpu_nmi;
   }

   if(this->measure_gpu_multichannel_nmi!=NULL)
   {
      this->measure_gpu_multichannel_nmi->InitialiseMeasure(this->currentReference,
            this->currentFloating,
            this->currentMask,
            this->activeVoxelNumber[this->currentLevel],
            this->warped,
            this->warpedGradientImage,
            this->voxelBasedMeasureGradientImage,
            &this->currentReference_gpu,
            &this->currentFloating_gpu,
            &this->currentMask_gpu,
            &this->warped_gpu,
            &this->warpedGradientImage_gpu,
            &this->voxelBasedMeasureGradientImage_gpu
                                                           );
      this->measure_multichannel_nmi=this->measure_gpu_multichannel_nmi;
   }

   if(this->measure_gpu_ssd!=NULL)
   {
      this->measure_gpu_ssd->InitialiseMeasure(this->currentReference,
            this->currentFloating,
            this->currentMask,
            this->activeVoxelNumber[this->currentLevel],
            this->warped,
            this->warpedGradientImage,
            this->voxelBasedMeasureGradientImage,
            &this->currentReference_gpu,
            &this->currentFloating_gpu,
            &this->currentMask_gpu,
            &this->warped_gpu,
            &this->warpedGradientImage_gpu,
            &this->voxelBasedMeasureGradientImage_gpu
                                              );
      this->measure_ssd=this->measure_gpu_ssd;
   }

   if(this->measure_gpu_kld!=NULL)
   {
      this->measure_gpu_kld->InitialiseMeasure(this->currentReference,
            this->currentFloating,
            this->currentMask,
            this->activeVoxelNumber[this->currentLevel],
            this->warped,
            this->warpedGradientImage,
            this->voxelBasedMeasureGradientImage,
            &this->currentReference_gpu,
            &this->currentFloating_gpu,
            &this->currentMask_gpu,
            &this->warped_gpu,
            &this->warpedGradientImage_gpu,
            &this->voxelBasedMeasureGradientImage_gpu
                                              );
      this->measure_kld=this->measure_gpu_kld;
   }

   if(this->measure_gpu_lncc!=NULL)
   {
      this->measure_gpu_lncc->InitialiseMeasure(this->currentReference,
            this->currentFloating,
            this->currentMask,
            this->activeVoxelNumber[this->currentLevel],
            this->warped,
            this->warpedGradientImage,
            this->voxelBasedMeasureGradientImage,
            &this->currentReference_gpu,
            &this->currentFloating_gpu,
            &this->currentMask_gpu,
            &this->warped_gpu,
            &this->warpedGradientImage_gpu,
            &this->voxelBasedMeasureGradientImage_gpu
                                               );
      this->measure_lncc=this->measure_gpu_lncc;
   }

   if(this->measure_gpu_dti!=NULL)
   {
      this->measure_gpu_dti->InitialiseMeasure(this->currentReference,
            this->currentFloating,
            this->currentMask,
            this->activeVoxelNumber[this->currentLevel],
            this->warped,
            this->warpedGradientImage,
            this->voxelBasedMeasureGradientImage,
            &this->currentReference_gpu,
            &this->currentFloating_gpu,
            &this->currentMask_gpu,
            &this->warped_gpu,
            &this->warpedGradientImage_gpu,
            &this->voxelBasedMeasureGradientImage_gpu
                                              );
      this->measure_dti=this->measure_gpu_dti;
   }
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_f3d_gpu::InitialiseSimilarity() done\n");
#endif
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif
