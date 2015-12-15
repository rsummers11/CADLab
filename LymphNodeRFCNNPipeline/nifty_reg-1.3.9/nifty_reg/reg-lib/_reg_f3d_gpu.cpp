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
template <class T>
reg_f3d_gpu<T>::reg_f3d_gpu(int refTimePoint,int floTimePoint)
    :reg_f3d<T>::reg_f3d(refTimePoint,floTimePoint)
{
    this->currentReference_gpu=NULL;
    this->currentFloating_gpu=NULL;
    this->currentMask_gpu=NULL;
    this->warped_gpu=NULL;
    this->controlPointGrid_gpu=NULL;
    this->deformationFieldImage_gpu=NULL;
    this->warpedGradientImage_gpu=NULL;
    this->voxelBasedMeasureGradientImage_gpu=NULL;
    this->nodeBasedGradientImage_gpu=NULL;
    this->conjugateG_gpu=NULL;
    this->conjugateH_gpu=NULL;
    this->bestControlPointPosition_gpu=NULL;
    this->logJointHistogram_gpu=NULL;

    this->currentReference2_gpu=NULL;
    this->currentFloating2_gpu=NULL;
    this->warped2_gpu=NULL;
    this->warpedGradientImage2_gpu=NULL;

    NR_CUDA_SAFE_CALL(cudaThreadExit())
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_gpu<T>::~reg_f3d_gpu()
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
    if(this->nodeBasedGradientImage_gpu!=NULL)
        cudaCommon_free<float4>(&this->nodeBasedGradientImage_gpu);
    if(this->conjugateG_gpu!=NULL)
        cudaCommon_free<float4>(&this->conjugateG_gpu);
    if(this->conjugateH_gpu!=NULL)
        cudaCommon_free<float4>(&this->conjugateH_gpu);
    if(this->bestControlPointPosition_gpu!=NULL)
        cudaCommon_free<float4>(&this->bestControlPointPosition_gpu);
    if(this->logJointHistogram_gpu!=NULL)
        cudaCommon_free<float>(&this->logJointHistogram_gpu);

    if(this->currentReference2_gpu!=NULL)
        cudaCommon_free(&this->currentReference2_gpu);
    if(this->currentFloating2_gpu!=NULL)
        cudaCommon_free(&this->currentFloating2_gpu);
    if(this->warped2_gpu!=NULL)
        cudaCommon_free<float>(&this->warped2_gpu);
    if(this->warpedGradientImage2_gpu!=NULL)
        cudaCommon_free<float4>(&this->warpedGradientImage2_gpu);

    NR_CUDA_SAFE_CALL(cudaThreadExit())
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateWarped()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateWarped called.\n");
#endif
    if(this->currentReference==NULL){
        printf("[NiftyReg ERROR] Error when allocating the warped image.\n");
        exit(1);
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
    if(this->warped->nt==1){
        if(cudaCommon_allocateArrayToDevice<float>(&this->warped_gpu, this->warped->dim)){
            printf("[NiftyReg ERROR] Error when allocating the warped image.\n");
            exit(1);
        }
    }
    else if(this->warped->nt==2){
        if(cudaCommon_allocateArrayToDevice<float>(&this->warped_gpu, &this->warped2_gpu, this->warped->dim)){
            printf("[NiftyReg ERROR] Error when allocating the warped image.\n");
            exit(1);
        }
    }
    else{
        printf("[NiftyReg ERROR] reg_f3d_gpu does not handle more than 2 time points in the floating image.\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateWarped done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearWarped()
{
    if(this->warped!=NULL){
        NR_CUDA_SAFE_CALL(cudaFreeHost(this->warped->data))
        this->warped->data = NULL;
        nifti_image_free(this->warped);
        this->warped=NULL;
    }
    if(this->warped_gpu!=NULL){
        cudaCommon_free<float>(&this->warped_gpu);
        this->warped_gpu=NULL;
    }
    if(this->warped2_gpu!=NULL){
        cudaCommon_free<float>(&this->warped2_gpu);
        this->warped2_gpu=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateDeformationField()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateDeformationField called.\n");
#endif
    this->ClearDeformationField();
    NR_CUDA_SAFE_CALL(cudaMalloc(&this->deformationFieldImage_gpu,
                                 this->activeVoxelNumber[this->currentLevel]*sizeof(float4)))

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateDeformationField done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearDeformationField()
{
    if(this->deformationFieldImage_gpu!=NULL){
        cudaCommon_free<float4>(&this->deformationFieldImage_gpu);
        this->deformationFieldImage_gpu=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateWarpedGradient()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateWarpedGradient called.\n");
#endif
    this->ClearWarpedGradient();
    if(this->inputFloating->nt==1){
        NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage_gpu,
                                     this->activeVoxelNumber[this->currentLevel]*sizeof(float4)))
    }
    else if(this->inputFloating->nt==2){
        NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage_gpu,
                                     this->activeVoxelNumber[this->currentLevel]*sizeof(float4)))
        NR_CUDA_SAFE_CALL(cudaMalloc(&this->warpedGradientImage2_gpu,
                                     this->activeVoxelNumber[this->currentLevel]*sizeof(float4)))
    }
    else{
        printf("[NiftyReg ERROR] reg_f3d_gpu does not handle more than 2 time points in the floating image.\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateWarpedGradient done.\n");
#endif

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearWarpedGradient()
{
    if(this->warpedGradientImage_gpu!=NULL){
        cudaCommon_free<float4>(&this->warpedGradientImage_gpu);
        this->warpedGradientImage_gpu=NULL;
    }
    if(this->warpedGradientImage2_gpu!=NULL){
        cudaCommon_free<float4>(&this->warpedGradientImage2_gpu);
        this->warpedGradientImage2_gpu=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateVoxelBasedMeasureGradient()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateVoxelBasedMeasureGradient called.\n");
#endif
    this->ClearVoxelBasedMeasureGradient();
    if(cudaCommon_allocateArrayToDevice(&this->voxelBasedMeasureGradientImage_gpu,
                                        this->currentReference->dim)){
        printf("[NiftyReg ERROR] Error when allocating the voxel based measure gradient image.\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateVoxelBasedMeasureGradient done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearVoxelBasedMeasureGradient()
{
    if(this->voxelBasedMeasureGradientImage_gpu!=NULL){
        cudaCommon_free<float4>(&this->voxelBasedMeasureGradientImage_gpu);
        this->voxelBasedMeasureGradientImage_gpu=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateNodeBasedGradient()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateNodeBasedGradient called.\n");
#endif
    this->ClearNodeBasedGradient();
    if(cudaCommon_allocateArrayToDevice(&this->nodeBasedGradientImage_gpu,
                                        this->controlPointGrid->dim)){
        printf("[NiftyReg ERROR] Error when allocating the node based gradient image.\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateNodeBasedGradient done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearNodeBasedGradient()
{
    if(this->nodeBasedGradientImage_gpu!=NULL){
        cudaCommon_free<float4>(&this->nodeBasedGradientImage_gpu);
        this->nodeBasedGradientImage_gpu=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateConjugateGradientVariables()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateConjugateGradientVariables called.\n");
#endif
    if(this->controlPointGrid==NULL){
        printf("[NiftyReg ERROR] Error when allocating the conjugate gradient arrays.\n");
        exit(1);
    }
    this->ClearConjugateGradientVariables();
    if(cudaCommon_allocateArrayToDevice(&this->conjugateG_gpu,
                                        this->controlPointGrid->dim)){
        printf("[NiftyReg ERROR] Error when allocating the conjugate gradient arrays.\n");
        exit(1);
    }
    if(cudaCommon_allocateArrayToDevice(&this->conjugateH_gpu,
                                        this->controlPointGrid->dim)){
        printf("[NiftyReg ERROR] Error when allocating the conjugate gradient arrays.\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateConjugateGradientVariables done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearConjugateGradientVariables()
{
    if(this->conjugateG_gpu!=NULL){
        cudaCommon_free<float4>(&this->conjugateG_gpu);
        this->conjugateG_gpu=NULL;
    }
    if(this->conjugateH_gpu!=NULL){
        cudaCommon_free<float4>(&this->conjugateH_gpu);
        this->conjugateH_gpu=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateBestControlPointArray()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateBestControlPointArray called.\n");
#endif
    if(this->controlPointGrid==NULL){
        printf("[NiftyReg ERROR] Error when allocating thebest control point array.\n");
        exit(1);
    }
    this->ClearBestControlPointArray();
    if(cudaCommon_allocateArrayToDevice(&this->bestControlPointPosition_gpu,
                                        this->controlPointGrid->dim)){
        printf("[NiftyReg ERROR] Error when allocating the best control point array.\n");
        exit(1);
    }
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateBestControlPointArray done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearBestControlPointArray()
{
    cudaCommon_free<float4>(&this->bestControlPointPosition_gpu);
    this->bestControlPointPosition_gpu=NULL;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateJointHistogram()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateJointHistogram called.\n");
#endif
    this->ClearJointHistogram();
    reg_f3d<T>::AllocateJointHistogram();
    NR_CUDA_SAFE_CALL(cudaMalloc(&this->logJointHistogram_gpu,
                                 this->totalBinNumber*sizeof(float)))
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateJointHistogram done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearJointHistogram()
{
    reg_f3d<T>::ClearJointHistogram();
    if(this->logJointHistogram_gpu!=NULL){
        cudaCommon_free<float>(&this->logJointHistogram_gpu);
        this->logJointHistogram_gpu=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::SaveCurrentControlPoint()
{
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->bestControlPointPosition_gpu, this->controlPointGrid_gpu,
                    this->controlPointGrid->nx*this->controlPointGrid->ny*
                    this->controlPointGrid->nz*sizeof(float4),
                    cudaMemcpyDeviceToDevice))
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::RestoreCurrentControlPoint()
{
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->controlPointGrid_gpu, this->bestControlPointPosition_gpu,
                    this->controlPointGrid->nx*this->controlPointGrid->ny*
                    this->controlPointGrid->nz*sizeof(float4),
                    cudaMemcpyDeviceToDevice))
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_gpu<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
    if(this->jacobianLogWeight<=0) return 0.;

    double value;
    if(type==2){
        value = reg_bspline_ComputeJacobianPenaltyTerm_gpu(this->currentReference,
                                                           this->controlPointGrid,
                                                           &this->controlPointGrid_gpu,
                                                           false);
    }
    else{
        value = reg_bspline_ComputeJacobianPenaltyTerm_gpu(this->currentReference,
                                                           this->controlPointGrid,
                                                           &this->controlPointGrid_gpu,
                                                           this->jacobianLogApproximation);
    }
    unsigned int maxit=5;
    if(type>0) maxit=20;
    unsigned int it=0;
    while(value!=value && it<maxit){
        if(type==2){
            value = reg_bspline_correctFolding_gpu(this->currentReference,
                                                   this->controlPointGrid,
                                                   &this->controlPointGrid_gpu,
                                                   false);
        }
        else{
            value = reg_bspline_correctFolding_gpu(this->currentReference,
                                                   this->controlPointGrid,
                                                   &this->controlPointGrid_gpu,
                                                   this->jacobianLogApproximation);
        }
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Folding correction\n");
#endif
        it++;
    }
    if(type>0){
        if(value!=value){
            this->RestoreCurrentControlPoint();
            fprintf(stderr, "[NiftyReg ERROR] The folding correction scheme failed\n");
        }
        else{
#ifdef NDEBUG
            if(this->verbose){
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
template <class T>
double reg_f3d_gpu<T>::ComputeBendingEnergyPenaltyTerm()
{
    if(this->bendingEnergyWeight<=0) return 0.;

    double value = reg_bspline_ApproxBendingEnergy_gpu(this->controlPointGrid,
                                                       &this->controlPointGrid_gpu);
    return this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::GetDeformationField()
{
    if(this->controlPointGrid_gpu==NULL){
        reg_f3d<T>::GetDeformationField();
    }
    else{
       // Compute the deformation field
        reg_bspline_gpu(this->controlPointGrid,
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
template <class T>
void reg_f3d_gpu<T>::WarpFloatingImage(int inter)
{
    // Interpolation is linear by default when using GPU, the inter variable is not used.
    inter=inter; // just to avoid a compiler warning

    // Compute the deformation field
    this->GetDeformationField();

    // Resample the floating image
    reg_resampleSourceImage_gpu(this->currentFloating,
                                &this->warped_gpu,
                                &this->currentFloating_gpu,
                                &this->deformationFieldImage_gpu,
                                &this->currentMask_gpu,
                                this->activeVoxelNumber[this->currentLevel],
                                this->warpedPaddingValue);
    if(this->currentFloating->nt==2){
        reg_resampleSourceImage_gpu(this->currentFloating,
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
template <class T>
double reg_f3d_gpu<T>::ComputeSimilarityMeasure()
{
    if(this->currentFloating->nt==1){
        if(cudaCommon_transferFromDeviceToNifti<float>
           (this->warped, &this->warped_gpu)){
            printf("[NiftyReg ERROR] Error when computing the similarity measure.\n");
            exit(1);
        }
    }
    else if(this->currentFloating->nt==2){
        if(cudaCommon_transferFromDeviceToNifti<float>
           (this->warped, &this->warped_gpu, &this->warped2_gpu)){
            printf("[NiftyReg ERROR] Error when computing the similarity measure.\n");
            exit(1);
        }
    }

    double measure=0.;
    if(this->currentFloating->nt==1){
        reg_getEntropies(this->currentReference,
                         this->warped,
                         this->referenceBinNumber,
                         this->floatingBinNumber,
                         this->probaJointHistogram,
                         this->logJointHistogram,
                         this->entropies,
                         this->currentMask,
                         this->approxParzenWindow);
    }
    else if(this->currentFloating->nt==2){
        reg_getEntropies2x2_gpu(this->currentReference,
                                 this->warped,
                                 //2,
                                 this->referenceBinNumber,
                                 this->floatingBinNumber,
                                 this->probaJointHistogram,
                                 this->logJointHistogram,
                                 &this->logJointHistogram_gpu,
                                 this->entropies,
                                 this->currentMask);
    }


    measure = double(this->entropies[0]+this->entropies[1])/double(this->entropies[2]);

    return double(1.0-this->bendingEnergyWeight-this->jacobianLogWeight) * measure;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::GetVoxelBasedGradient()
{
    // The log joint histogram is first transfered to the GPU
    float *tempB=NULL;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&tempB, this->totalBinNumber*sizeof(float)))
    for(unsigned int i=0; i<this->totalBinNumber;i++){
        tempB[i]=(float)this->logJointHistogram[i];
    }
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->logJointHistogram_gpu, tempB,
                                 this->totalBinNumber*sizeof(float), cudaMemcpyHostToDevice))
    NR_CUDA_SAFE_CALL(cudaFreeHost(tempB))

    // The intensity gradient is first computed
    reg_getSourceImageGradient_gpu( this->currentFloating,
                                    &this->currentFloating_gpu,
                                    &this->deformationFieldImage_gpu,
                                    &this->warpedGradientImage_gpu,
                                    this->activeVoxelNumber[this->currentLevel]);

    if(this->currentFloating->nt==2){
        reg_getSourceImageGradient_gpu( this->currentFloating,
                                        &this->currentFloating2_gpu,
                                        &this->deformationFieldImage_gpu,
                                        &this->warpedGradientImage2_gpu,
                                        this->activeVoxelNumber[this->currentLevel]);
    }

    // The voxel based NMI gradient
    if(this->currentFloating->nt==1){
        reg_getVoxelBasedNMIGradientUsingPW_gpu(this->currentReference,
                                                this->warped,
                                                &this->currentReference_gpu,
                                                &this->warped_gpu,
                                                &this->warpedGradientImage_gpu,
                                                &this->logJointHistogram_gpu,
                                                &this->voxelBasedMeasureGradientImage_gpu,
                                                &this->currentMask_gpu,
                                                this->activeVoxelNumber[this->currentLevel],
                                                this->entropies,
                                                this->referenceBinNumber[0],
                                                this->floatingBinNumber[0]);
    }
    else if(this->currentFloating->nt==2){
        reg_getVoxelBasedNMIGradientUsingPW2x2_gpu( this->currentReference,
                                                    this->warped,
                                                    &this->currentReference_gpu,
                                                    &this->currentReference2_gpu,
                                                    &this->warped_gpu,
                                                    &this->warped2_gpu,
                                                    &this->warpedGradientImage_gpu,
                                                    &this->warpedGradientImage2_gpu,
                                                    &this->logJointHistogram_gpu,
                                                    &this->voxelBasedMeasureGradientImage_gpu,
                                                    &this->currentMask_gpu,
                                                    this->activeVoxelNumber[this->currentLevel],
                                                    this->entropies,
                                                    this->referenceBinNumber,
                                                    this->floatingBinNumber);
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::GetSimilarityMeasureGradient()
{

    this->GetVoxelBasedGradient();

    // The voxel based gradient is smoothed
    int smoothingRadius[3];
    smoothingRadius[0] = (int)( 2.0*this->controlPointGrid->dx/this->currentReference->dx );
    smoothingRadius[1] = (int)( 2.0*this->controlPointGrid->dy/this->currentReference->dy );
    smoothingRadius[2] = (int)( 2.0*this->controlPointGrid->dz/this->currentReference->dz );
    reg_smoothImageForCubicSpline_gpu(  this->warped,
                                        &this->voxelBasedMeasureGradientImage_gpu,
                                        smoothingRadius);
    // The node gradient is extracted
    reg_voxelCentric2NodeCentric_gpu(   this->warped,
                                        this->controlPointGrid,
                                        &this->voxelBasedMeasureGradientImage_gpu,
                                        &this->nodeBasedGradientImage_gpu,
                                        1.0-this->bendingEnergyWeight-this->jacobianLogWeight);
    /* The NMI gradient is converted from voxel space to real space */
    mat44 *floatingMatrix_xyz=NULL;
    if(this->currentFloating->sform_code>0)
        floatingMatrix_xyz = &(this->currentFloating->sto_xyz);
    else floatingMatrix_xyz = &(this->currentFloating->qto_xyz);
    reg_convertNMIGradientFromVoxelToRealSpace_gpu( floatingMatrix_xyz,
                                                    this->controlPointGrid,
                                                    &this->nodeBasedGradientImage_gpu);
    // The gradient is smoothed using a Gaussian kernel if it is required
    if(this->gradientSmoothingSigma!=0){
        reg_gaussianSmoothing_gpu(this->controlPointGrid,
                                  &this->nodeBasedGradientImage_gpu,
                                  this->gradientSmoothingSigma,
                                  NULL);
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::GetBendingEnergyGradient()
{
    if(this->bendingEnergyWeight<=0) return;

    reg_bspline_ApproxBendingEnergyGradient_gpu(this->currentReference,
                                                 this->controlPointGrid,
                                                 &this->controlPointGrid_gpu,
                                                 &this->nodeBasedGradientImage_gpu,
                                                 this->bendingEnergyWeight);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::GetJacobianBasedGradient()
{
    if(this->jacobianLogWeight<=0) return;

    reg_bspline_ComputeJacobianPenaltyTermGradient_gpu(this->currentReference,
                                                       this->controlPointGrid,
                                                       &this->controlPointGrid_gpu,
                                                       &this->nodeBasedGradientImage_gpu,
                                                       this->jacobianLogWeight,
                                                       this->jacobianLogApproximation);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ComputeConjugateGradient()
{
    if(this->currentIteration==1){
        // first conjugate gradient iteration
        reg_initialiseConjugateGradient(&this->nodeBasedGradientImage_gpu,
                                        &this->conjugateG_gpu,
                                        &this->conjugateH_gpu,
                                        this->controlPointGrid->nx*this->controlPointGrid->ny*this->controlPointGrid->nz);
    }
    else{
        // conjugate gradient computation if iteration != 1
        reg_GetConjugateGradient(&this->nodeBasedGradientImage_gpu,
                                 &this->conjugateG_gpu,
                                 &this->conjugateH_gpu,
                                 this->controlPointGrid->nx*
                                 this->controlPointGrid->ny*
                                 this->controlPointGrid->nz);
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d_gpu<T>::GetMaximalGradientLength()
{
    T maxLength = reg_getMaximalLength_gpu(&this->nodeBasedGradientImage_gpu,
                                           this->controlPointGrid->nx*
                                           this->controlPointGrid->ny*
                                           this->controlPointGrid->nz);
    return maxLength;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::UpdateControlPointPosition(T scale)
{
    reg_updateControlPointPosition_gpu(this->controlPointGrid,
                                       &this->controlPointGrid_gpu,
                                       &this->bestControlPointPosition_gpu,
                                       &this->nodeBasedGradientImage_gpu,
                                       scale);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::AllocateCurrentInputImage()
{
    reg_f3d<T>::AllocateCurrentInputImage();

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateCurrentInputImage called.\n");
#endif

    if(this->currentReference_gpu!=NULL) cudaCommon_free(&this->currentReference_gpu);
    if(this->currentReference2_gpu!=NULL) cudaCommon_free(&this->currentReference2_gpu);
    if(this->currentReference->nt==1){
        if(cudaCommon_allocateArrayToDevice<float>
           (&this->currentReference_gpu, this->currentReference->dim)){
            printf("[NiftyReg ERROR] Error when allocating the reference image.\n");
            exit(1);
        }
        if(cudaCommon_transferNiftiToArrayOnDevice<float>
           (&this->currentReference_gpu, this->currentReference)){
            printf("[NiftyReg ERROR] Error when transfering the reference image.\n");
            exit(1);
        }
    }
    else if(this->currentReference->nt==2){
        if(cudaCommon_allocateArrayToDevice<float>
           (&this->currentReference_gpu,&this->currentReference2_gpu, this->currentReference->dim)){
            printf("[NiftyReg ERROR] Error when allocating the reference image.\n");
            exit(1);
        }
        if(cudaCommon_transferNiftiToArrayOnDevice<float>
           (&this->currentReference_gpu, &this->currentReference2_gpu, this->currentReference)){
            printf("[NiftyReg ERROR] Error when transfering the reference image.\n");
            exit(1);
        }
    }

    if(this->currentFloating_gpu!=NULL) cudaCommon_free(&this->currentFloating_gpu);
    if(this->currentFloating2_gpu!=NULL) cudaCommon_free(&this->currentFloating2_gpu);
    if(this->currentReference->nt==1){
        if(cudaCommon_allocateArrayToDevice<float>
           (&this->currentFloating_gpu, this->currentFloating->dim)){
            printf("[NiftyReg ERROR] Error when allocating the floating image.\n");
            exit(1);
        }
        if(cudaCommon_transferNiftiToArrayOnDevice<float>
           (&this->currentFloating_gpu, this->currentFloating)){
            printf("[NiftyReg ERROR] Error when transfering the floating image.\n");
            exit(1);
        }
    }
    else if(this->currentReference->nt==2){
        if(cudaCommon_allocateArrayToDevice<float>
           (&this->currentFloating_gpu, &this->currentFloating2_gpu, this->currentFloating->dim)){
            printf("[NiftyReg ERROR] Error when allocating the floating image.\n");
            exit(1);
        }
        if(cudaCommon_transferNiftiToArrayOnDevice<float>
           (&this->currentFloating_gpu, &this->currentFloating2_gpu, this->currentFloating)){
            printf("[NiftyReg ERROR] Error when transfering the floating image.\n");
            exit(1);
        }
    }
    if(this->controlPointGrid_gpu!=NULL) cudaCommon_free<float4>(&this->controlPointGrid_gpu);
    if(cudaCommon_allocateArrayToDevice<float4>
       (&this->controlPointGrid_gpu, this->controlPointGrid->dim)){
        printf("[NiftyReg ERROR] Error when allocating the control point image.\n");
        exit(1);
    }

    if(cudaCommon_transferNiftiToArrayOnDevice<float4>
       (&this->controlPointGrid_gpu, this->controlPointGrid)){
        printf("[NiftyReg ERROR] Error when transfering the control point image.\n");
        exit(1);
    }

    int *targetMask_h;
    NR_CUDA_SAFE_CALL(cudaMallocHost(&targetMask_h,this->activeVoxelNumber[this->currentLevel]*sizeof(int)))
    int *targetMask_h_ptr = &targetMask_h[0];
    for(int i=0;i<this->currentReference->nx*this->currentReference->ny*this->currentReference->nz;i++){
        if( this->currentMask[i]!=-1) *targetMask_h_ptr++=i;
    }
    NR_CUDA_SAFE_CALL(cudaMalloc(&this->currentMask_gpu,
                                 this->activeVoxelNumber[this->currentLevel]*sizeof(int)))
    NR_CUDA_SAFE_CALL(cudaMemcpy(this->currentMask_gpu, targetMask_h,
                                 this->activeVoxelNumber[this->currentLevel]*sizeof(int),
                                 cudaMemcpyHostToDevice))
    NR_CUDA_SAFE_CALL(cudaFreeHost(targetMask_h))
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::AllocateCurrentInputImage done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_gpu<T>::ClearCurrentInputImage()
{
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::ClearCurrentInputImage called.\n");
#endif
    if(cudaCommon_transferFromDeviceToNifti<float4>
       (this->controlPointGrid, &this->controlPointGrid_gpu)){
        printf("[NiftyReg ERROR] Error when transfering back the control point image.\n");
        exit(1);
    }
    cudaCommon_free<float4>(&this->controlPointGrid_gpu);
    this->controlPointGrid_gpu=NULL;
    cudaCommon_free(&this->currentReference_gpu);
    this->currentReference_gpu=NULL;
    cudaCommon_free(&this->currentFloating_gpu);
    this->currentFloating_gpu=NULL;
    NR_CUDA_SAFE_CALL(cudaFree(this->currentMask_gpu))
    this->currentMask_gpu=NULL;

    if(this->currentReference->nt==2){
        cudaCommon_free(&this->currentReference2_gpu);
        this->currentReference2_gpu=NULL;
        cudaCommon_free(&this->currentFloating2_gpu);
        this->currentFloating2_gpu=NULL;
    }
    this->currentReference=NULL;
    this->currentMask=NULL;
    this->currentFloating=NULL;
#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_gpu<T>::ClearCurrentInputImage done.\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
int reg_f3d_gpu<T>::CheckMemoryMB_f3d()
{
    if(!this->initialised) reg_f3d<T>::Initisalise_f3d();

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

    // joint histogram
    unsigned int histogramSize[3]={1,1,1};
    for(int i=0;i<this->referenceTimePoint;i++){
        histogramSize[0] *= this->referenceBinNumber[i];
        histogramSize[1] *= this->referenceBinNumber[i];
    }
    for(int i=0;i<this->floatingTimePoint;i++){
        histogramSize[0] *= this->floatingBinNumber[i];
        histogramSize[2] *= this->floatingBinNumber[i];
    }
    histogramSize[0] += histogramSize[1] + histogramSize[2];
    totalMemoryRequiered += histogramSize[0] * sizeof(float);

    // jacobian array
    if(this->jacobianLogWeight>0)
        totalMemoryRequiered += 10 * referenceVoxelNumber *
                                sizeof(float);

    return (int)(ceil((float)totalMemoryRequiered/float(1024*1024)));

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
